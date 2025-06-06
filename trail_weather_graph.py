# trail_weather_graph.py
"""
Conversation flow
-----------------
1️⃣ parse_user_input   – extract `location`, `activity`, and `desired features`
2️⃣ find_trails        – TrailAgent → DataFrame of nearby trails
3️⃣ select_trail       – pick the best candidate (rank 0, 1, 2…)
4️⃣ check_weather      – WeatherAgent → current weather at that trail
5️⃣ evaluate_weather   – if "good", continue, else loop back to 3️⃣
6️⃣ compose_response   – craft the final answer for the user
"""

import ast
import asyncio
import json
import os
from typing import Annotated, Dict, List, Literal, TypedDict

import geocoder
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity

import utils
from agents.google_maps import GoogleMaps, GoogleMapsAgent
from agents.trail_agent import SheetsAgent, SheetsResponse
from agents.weather_agent import Weather, WeatherAgent

# ──────────────────────────────  INIT  ──────────────────────────────
load_dotenv()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

sheets_agent = SheetsAgent()
google_maps_agent = GoogleMapsAgent(google_api_key=GOOGLE_API_KEY)
claude_llm = init_chat_model(model="claude-sonnet-4-20250514")

feature_vector_cache: dict[str, np.ndarray] = {}


# ───────────────────────  STATE & SCHEMAS  ─────────────────────────


class ParsedInput(BaseModel):
    """
    ParsedInput is a Pydantic model that is used to parse the user's input.
    It is used to extract the user's location, activity, and desired features.
    - location: str | None  # "Boston, MA"
    - activities: List[str] | None  # e.g. ["hiking", "birding"]
    - features: List[str] | None  # e.g. ["lake", "forest"]
    """

    location: str = Field(..., description="City-state or place name")
    activities: List[str] = Field(
        default_factory=list,
        description="""Optional list of outdoor activities of interest such as 
        walking, backpacking, canoeing, skiing, paddle-sports, camping, 
        fly-fishing, mountain-biking, hiking, road-biking, birding, snowboarding, 
        horseback-riding, scenic-driving, cross-country-skiing, rails-trails, 
        snowshoeing, whitewater-kayaking, off-road-driving, nature-trips, fishing, 
        sea-kayaking, rock-climbing, trail-running, surfing, bike-touring, 
        ice-climbing""",
    )
    features: List[str] = Field(
        default_factory=list,
        description="""Optional list of desired trail features such as wildlife, 
        paved, waterfall, historic-site, dogs-leash, wild-flowers, beach, 
        rails-trails, views, cave, partially-paved, forest, kids, river, 
        strollers, lake, dogs, hot-springs, dogs-no, ada, city-walk""",
    )


class State(TypedDict):
    messages: Annotated[list, add_messages]  # full chat transcript
    parsed_input: ParsedInput | None  # parsed input from user
    trails_df: pd.DataFrame | None  # all candidate trails
    trail_idx: int | None  # pointer into trails_df
    current_trail: SheetsResponse | None  # chosen trail dataclass
    weather: Weather | None  # weather for trail
    weather_ok: bool | None  # ✔︎ / ✘ decision flag
    message_type: str | None  # message type


class WeatherDecision(BaseModel):
    """
    WeatherDecision is a Pydantic model that is used to decide if the weather is good enough for the activity.
    It is used to determine if the weather is suitable for the activity.
    - weather_ok: bool | None  # ✔︎ / ✘ decision flag
    """

    weather_ok: Literal[True, False] = Field(
        ..., description="True if conditions are suitable for the activity"
    )
    reason: str = Field(..., description="Reason for the weather decision")


class MessageClassifier(BaseModel):
    message_type: Literal["parse", "other"] = Field(
        ...,
        description="""The type of message. If the message contains a location, 
        activity and any trail features, return 'parse'. If the message does not 
        contain a location or activity, return 'other'.""",
    )


class PandasAgentResult(BaseModel):
    """
    PandasAgentResult is a Pydantic model that is used to parse the agent's result.
    It is used to parse the agent's result into a pandas dataframe.
    """

    name: str
    area_name: str
    distance: float
    activities: List[str]
    features: List[str]
    lat: float
    lng: float


# ────────────────────────────  NODES  ──────────────────────────────


def classifier(state: State) -> dict:
    last_message = state["messages"][-1]
    classifier_llm = claude_llm.with_structured_output(MessageClassifier)
    result = classifier_llm.invoke(
        [
            {
                "role": "system",
                "content": """You are a message classifier. Classify the message as either:
                - "parse": if the message contains a location or activity or any trail features
                - "other": if the message does not contain a location or activity or any trail features
                Return exactly "parse" or "other" in lowercase.
                """,
            },
            {
                "role": "user",
                "content": last_message.content,
            },
        ]
    )
    return {"message_type": result.message_type}


def router(state: State) -> dict:
    message_type = state.get("message_type", "parse")
    if message_type == "parse":
        return {"next": "parse_user_input"}
    return {"next": "other_llm_response"}


def other_llm_response(state: State) -> dict:
    last_message = state["messages"][-1]
    result = claude_llm.invoke(
        [
            {
                "role": "system",
                "content": """You are a helpful assistant. You are given a 
                message and you need to return a response to the message. The 
                response should be a single line. The response should be 
                in a friendly and engaging tone.""",
            },
            {
                "role": "user",
                "content": last_message.content,
            },
        ]
    )
    return {"messages": [{"role": "assistant", "content": result.content}]}


def parse_user_input(state: State) -> Dict:
    """
    Use Claude to pull out the user's location and desired activity.
    Expects the *latest* user message to contain both.
    """
    last = state["messages"][-1]
    extractor = claude_llm.with_structured_output(ParsedInput)
    result = extractor.invoke(
        [
            {
                "role": "system",
                "content": (
                    """You are a parsing assistant. From the following message, 
                    return:
                    - the user's LOCATION (city + state or landmark), 
                    - preferred OUTDOOR ACTIVITIES: what activities a user can do on the trail (can contain multiple activities, return an empty list if none),
                    - preferred TRAIL FEATURES: what features the user wants to see on the trail (can contain multiple features, return an empty list if none).
                    If the user does not mention a location, activities or 
                    features, return None for that field.
                    
                    Example:
                    User: I'm in Boston, MA and I want to go hiking and birding. I also want to see a waterfall and wildlife (if possible).
                    You: location: Boston, MA, features: ['hiking', 'birding'], activities: ['waterfall', 'wildlife']

                    Example:
                    User: I'm in Boston, MA and I want to go fishing. I want to find a trail which allows dogs.
                    You: location: Boston, MA, activities: ['fishing'], features: ['dogs', 'dogs-leash']
                    """
                ),
            },
            {"role": "user", "content": last.content},
        ]
    )
    print(f"Parsed input: {result}")
    return {"parsed_input": result}


def compute_similarity(
    row: pd.Series,
    desired_activities: List[str],
    desired_feats: List[str],
    feature_vector_cache: dict[str, np.ndarray],
):
    # Similar to trail_matches, but returns the max similarity score instead of just True/False
    trail_feats = [f.lower() for f in row.features]
    trail_acts = [a.lower() for a in row.activities]
    token_vectors = np.vstack(
        [
            utils.get_feature_vector(t, feature_vector_cache)
            for t in trail_feats + trail_acts
        ]
    )
    scores = []
    for act in desired_activities:
        sim = cosine_similarity(
            utils.get_feature_vector(act, feature_vector_cache).reshape(1, -1),
            token_vectors,
        ).max()
        scores.append(sim)
    for feat in desired_feats:
        sim = cosine_similarity(
            utils.get_feature_vector(feat, feature_vector_cache).reshape(1, -1),
            token_vectors,
        ).max()
        scores.append(sim)
    # Return average or min similarity as the score
    return np.mean(scores) if scores else 0.0


def get_distances_in_batches(
    source: List[float],
    destinations: List[str],
    batch_size: int = 25,
) -> List[GoogleMaps]:
    print(
        f"Getting distances for {len(destinations)} destinations in batches of {batch_size}"
    )
    all_results = []
    for i in range(0, len(destinations), batch_size):
        batch = destinations[i : i + batch_size]
        result = google_maps_agent.get_directions(source, batch)
        all_results.extend(result.maps_df)
    print(f"Got {len(all_results)} distances")
    return all_results


def rank_trails_with_agent(
    trails_df: pd.DataFrame,
    user_location: List[float],
    desired_activities: List[str],
    desired_features: List[str],
):
    pandas_agent = create_pandas_dataframe_agent(
        llm=claude_llm,
        df=trails_df,
        verbose=True,
        allow_dangerous_code=True,
        max_execution_time=120,  # number of seconds to run the code before stopping
        max_iterations=40,  # number of times to retry if the code fails
        # agent_type="openai-functions",
    )
    query = f"""
        Rank the trails by best match for activities {desired_activities} and features {desired_features},
        closest to {user_location} (use the 'distance' column for proximity).
        Return the top 5 trails with their name, area_name, distance, activities, features, lat, lng.
        The response should be in this format:
        [
            {{"name": "a", "area_name": "b", "distance": "100 mi", "activities": ["c", "x"], "features": ["d", "p"], "lat": 1.1, "lng": 1.1}},
            {{"name": "e", "area_name": "f", "distance": "2,200 mi", "activities": ["g", "y"], "features": ["h", "q", "s"], "lat": 2.2, "lng": 2.2}},
            {{"name": "i", "area_name": "j", "distance": None, "activities": ["k", "x", "z"], "features": ["l", "r"], "lat": 3.3, "lng": 3.3}}
        ]
        Do not modify the 'distance' column (if you need to convert it to a number, do it in the code only). Do not include any other columns in the dataframe. Do not include any other text in the response.
    """
    return pandas_agent.invoke(query)
    # return pandas_agent.run(query)


def find_trails(state: State):
    """
    Call TrailAgent once, getting *all* trails sorted by distance,
    and keep them in the state DataFrame.
    """

    user_location = state.get("parsed_input").location  # "Boston, MA"
    g = geocoder.google(user_location, key=GOOGLE_API_KEY)
    if not g.latlng:
        return {"trails_df": None, "trail_idx": None}
    user_location = g.latlng  # (lat, lng)

    # Get all trails from the sheet
    trails = sheets_agent.get_sheet()  # returns TrailResponse list
    trails_df = pd.DataFrame([t.model_dump() for t in trails.sheet_df])

    # Convert lat and lng to create a destination string
    trails_df["destination_str"] = (
        trails_df["lat"].astype(str) + "," + trails_df["lng"].astype(str)
    )
    destinations = trails_df["destination_str"].tolist()
    distance_results = get_distances_in_batches(user_location, destinations)

    # Map trail name to distance
    name_to_distance = {}
    for result in distance_results:
        # for dest, dist in zip(result.destinations_latlng, result.distance):
        name_to_distance[result.destinations_latlng] = result.distance

    trails_df["distance"] = trails_df["destination_str"].map(name_to_distance)
    # print(trails_df.head())

    # # Save the trails_df to a csv file
    # trails_df.to_csv("trails_df_distances.csv", index=False)

    # trails_df = pd.read_csv("trails_df_distances.csv")

    # Further filter by desired features, if any
    desired_feats = state.get("parsed_input").features
    desired_acts = state.get("parsed_input").activities

    if desired_acts or desired_feats:
        try:
            # Use LLM agent to rank and select top 5 trails
            agent_result_str = rank_trails_with_agent(
                trails_df,
                user_location,
                desired_acts,
                desired_feats,
            )
            print(f"Agent result {type(agent_result_str)}: {agent_result_str}")
            # Optionally, parse agent_result to a DataFrame or list for downstream use
            # For now, just return the agent result in the state
            try:
                if isinstance(agent_result_str, dict):
                    agent_result = agent_result_str.get("output", "")
                    if isinstance(agent_result, str):
                        try:
                            agent_result = json.loads(agent_result)
                        except Exception:
                            agent_result = ast.literal_eval(agent_result)
                elif isinstance(agent_result_str, str):
                    try:
                        agent_result = json.loads(agent_result_str)
                    except Exception:
                        agent_result = ast.literal_eval(agent_result_str)
                elif isinstance(agent_result_str, list):
                    agent_result = [
                        ast.literal_eval(x) if isinstance(x, str) else x
                        for x in agent_result_str
                    ]
                else:
                    raise ValueError("Invalid agent result")

                agent_df = pd.DataFrame(agent_result)
                # Ensure activities/features are lists, not strings
                for col in ["activities", "features"]:
                    agent_df[col] = agent_df[col].apply(
                        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                    )
            except Exception as e:
                print("Error parsing agent result", e)
                agent_df = trails_df.head(5).reset_index(drop=True)

            # print(f"Agent df dtypes (before merge): {agent_df.dtypes}")
            # print(f"Trails df dtypes: {trails_df.dtypes}")

            for col in ["distance", "lat", "lng"]:
                agent_df[col] = pd.to_numeric(agent_df[col], errors="coerce")
                trails_df[col] = pd.to_numeric(trails_df[col], errors="coerce")

            for col in ["activities", "features"]:
                # agent_df[col] = agent_df[col].apply(
                #     lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                # )
                trails_df[col] = trails_df[col].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                )

            # Merge trails_df with agent_df on the name column
            agent_df = pd.merge(
                agent_df,
                trails_df,
                on=[
                    "name",
                    "area_name",
                    "distance",
                    "lat",
                    "lng",
                ],
                how="inner",
            )

            # print(f"Agent df dtypes (after merge): {agent_df.dtypes}")

            agent_df = agent_df.drop(columns=["activities_x", "features_x"])
            agent_df = agent_df.rename(
                columns={"activities_y": "activities", "features_y": "features"}
            )

            print(f"Agent df: {agent_df.head()}")

            return {
                "trails_df": agent_df,
                "trail_idx": 0,
                # "agent_result": agent_result,
            }
        except Exception as e:
            print("Error finding trails", e)
            return {"trails_df": trails_df, "trail_idx": 0}
    else:
        trails_df = trails_df.head(5).reset_index(drop=True)
        return {"trails_df": trails_df, "trail_idx": 0}


def select_trail(state: State):
    """
    Pick the next candidate trail based on `trail_idx`.
    """

    df: pd.DataFrame = state.get("trails_df")
    idx = state.get("trail_idx")

    if idx >= len(df):
        # Ran out of options
        return {
            "current_trail": None,
            "weather": None,
            "weather_ok": False,
            "trail_idx": idx,
        }

    row = df.iloc[idx]
    trail_obj = SheetsResponse(**row.to_dict())
    # print(f"Selected trail: {trail_obj.name}")
    return {"current_trail": trail_obj}


def check_weather(state: State):
    """
    Query WeatherAgent for the current conditions *at the trail*.
    """
    trail: SheetsResponse = state.get("current_trail")
    if trail is None:
        return {"weather": None}

    weather_agent = WeatherAgent(
        city=trail.city_name,
        state=trail.state_name,
        openweather_api_key=OPENWEATHER_API_KEY,
    )
    weather = asyncio.run(weather_agent.get_weather(trail.lat, trail.lng))
    print(
        f"Weather: The weather at {trail.name} is {weather.condition} "
        f"({weather.condition_description}) with temperature "
        f"{weather.temperature} and feels like {weather.feels_like}"
    )
    return {"weather": weather}


def evaluate_weather(state: State):
    """
    Decide if the weather is good enough for the activity.
    A very simple rule: no rain/snow + temp between 50–85 °F.
    Refine as needed.
    """
    weather: Weather = state.get("weather")
    # If no weather data is available (e.g., we have exhausted all trail candidates),
    # stop the search loop and proceed to compose a "no suitable trail" response.
    if weather is None:
        return {"weather_ok": True}

    good_temp = 50 <= weather.temperature <= 85
    good_cond = weather.condition.lower() not in [
        "rain",
        "snow",
        "thunderstorm",
        "clouds",
    ]
    return {"weather_ok": good_temp and good_cond}


def increment_index(state: State):
    """
    Advance to the next trail candidate.
    """
    return {"trail_idx": state.get("trail_idx") + 1}


def compose_alterate_response(trail, trails_df) -> str:
    # Find the best trail row in trails_df
    best_idx = trails_df[
        (trails_df["name"] == trail.name)
        & (trails_df["area_name"] == trail.area_name)
        & (trails_df["lat"] == trail.lat)
        & (trails_df["lng"] == trail.lng)
    ].index

    # Exclude the best trail to get alternatives
    alternatives_df = trails_df.drop(best_idx).reset_index(drop=True)

    alt_msgs = []

    # Format alternatives
    alt_msgs = []
    for idx, row in alternatives_df.iterrows():
        alt_msgs.append(
            f"**Alternative {idx+1}:** {row['name']} in {row['area_name']}, {row['city_name']}, {row['state_name']} - {row['distance']} away. "
            f"Features: {', '.join(row['features'])}. Activities: {', '.join(row['activities'])}."
        )
    alt_msg = "\n\n".join(alt_msgs)

    return alt_msg


def compose_response(state: State):
    """
    Combine trail + weather into a nice human reply.
    """
    trail: SheetsResponse = state.get("current_trail")
    weather: Weather = state.get("weather")
    trails_df = state.get("trails_df")

    # Save the trails_df to a csv file
    trails_df.to_csv("trails_df.csv", index=False)

    # Convert the trails_df to a list of TrailResponse objects
    trails_list = [SheetsResponse(**row) for row in trails_df.to_dict("records")]

    extractor = claude_llm
    if trail is None:
        msg = (
            "Sorry, I couldn't find any trail near you with suitable weather "
            f"for {state.get('parsed_input').activities} or "
            f"{state.get('parsed_input').features} right now that have a good "
            "weather. Please try again with different criteria."
        )
    else:
        distance = trails_df.distance.iloc[0]
        # Convert trail length to miles based on trail_units
        trail_length = (
            f"{trail.length * 0.000189394:.2f} mi"
            if trail.units == "i"
            else f"{trail.length * 0.000621371:.2f} mi"
            if trail.units == "m"
            else f"{trail.length} mi"
        )
        msg = (
            f"**Best match:**\n{trail.name} in {trail.area_name}, {trail.city_name}, {trail.state_name} is {distance} away."
            f"Length of the trail is {trail_length} with trail type of {trail.route_type}, difficulty rated at {trail.difficulty_rating}."
            f"Average rating is {trail.avg_rating} with {trail.num_reviews} reviews.\n"
            f"Features: {', '.join(trail.features)}.\n"
            f"Activities supported: {', '.join(trail.activities)}.\n\n"
        )

        # Mention whether all requested features were satisfied
        desired_feats = state.get("parsed_input").features
        desired_activities = state.get("parsed_input").activities
        if desired_feats:
            missing = [
                f
                for f in desired_feats
                if f not in [feat.lower() for feat in trail.features]
            ]
            if missing:
                msg += f"""Note: This trail is missing these requested features:
                {', '.join(missing)}.\n\n"""
            else:
                msg += "All your requested features are present! 🎉\n\n"
        elif desired_activities:
            missing = [
                a
                for a in desired_activities
                if a not in [act.lower() for act in trail.activities]
            ]
            if missing:
                msg += f"""Note: This trail is missing these requested activities:
                {', '.join(missing)}.\n\n"""
            else:
                msg += "All your requested activities are present! 🎉\n\n"
        else:
            msg += "\n"

        if weather is not None:
            msg += (
                f"**Weather at trailhead:**\n Weather in {weather.city}, "
                f"{weather.state} is {weather.condition} "
                f"({weather.condition_description}). The temperature is "
                f"{weather.temperature} °F (feels {weather.feels_like} °F), "
                f"with winds up to {weather.wind} mph and humidity of "
                f"{weather.humidity}%.\n\n"
            )
        else:
            msg += "Weather data is currently unavailable.\n\n"
    #     msg += "Have a great outing!"

    alt_msg = compose_alterate_response(trail, trails_df)
    msg += f"\n\nOther trails that might interest you:\n{alt_msg}"

    result = extractor.invoke(
        [
            {
                "role": "system",
                "content": (
                    """You are an expert in analyzing trail data in dataframe.
                    You are also given the weather and trail info of the best match.
                    You are also given a list of alternate trails that might interest the user.
                    You need to create a response that summarises the weather at
                    the trail (locaiton, temperature, condition, etc.) and the
                    trail (name, area name, popularity, length, difficulty, etc.)
                    """
                ),
            },
            {
                "role": "user",
                "content": f"""Best match response containing trail info and weather: {msg}
                All trails: {trails_list}
                
                For alternate trails, mention the trail name, area name, distance, activities, features.
                
                The response should be in a engaging and friendly tone and should include appropriate emojis.
                """,
            },
        ]
    )
    return {
        "messages": [
            {
                "role": "assistant",
                "content": result.content,
            }
        ]
    }


# ───────────────────────────  GRAPH  ───────────────────────────────
def build_graph():
    graph_builder = StateGraph(State)

    graph_builder.add_node("classifier", classifier)
    graph_builder.add_node("router", router)
    graph_builder.add_node("other_llm_response", other_llm_response)
    graph_builder.add_node("parse_user_input", parse_user_input)
    graph_builder.add_node("find_trails", find_trails)
    graph_builder.add_node("select_trail", select_trail)
    graph_builder.add_node("check_weather", check_weather)
    graph_builder.add_node("evaluate_weather", evaluate_weather)
    graph_builder.add_node("increment_index", increment_index)
    graph_builder.add_node("compose_response", compose_response)

    # Edges
    graph_builder.add_edge(START, "classifier")
    graph_builder.add_edge("classifier", "router")
    # Conditional edges
    graph_builder.add_conditional_edges(
        "router",
        lambda state: state.get("next"),
        {
            "parse_user_input": "parse_user_input",
            "other_llm_response": "other_llm_response",
        },
    )

    # graph_builder.add_edge("router", "parse_user_input")
    graph_builder.add_edge("parse_user_input", "find_trails")
    graph_builder.add_edge("find_trails", "select_trail")
    graph_builder.add_edge("select_trail", "check_weather")
    graph_builder.add_edge("check_weather", "evaluate_weather")

    # Conditional edge: good weather?
    graph_builder.add_conditional_edges(
        "evaluate_weather",
        lambda s: "good" if s["weather_ok"] else "bad",
        {
            "good": "compose_response",
            "bad": "increment_index",
        },
    )

    # If weather bad → next trail → weather again
    graph_builder.add_edge("increment_index", "select_trail")
    graph_builder.add_edge("select_trail", "compose_response")

    # Success
    graph_builder.add_edge("compose_response", END)
    graph_builder.add_edge("other_llm_response", END)

    return graph_builder.compile()


# ───────────────────────────  CLI DEMO  ────────────────────────────
def run_chatbot():
    graph = build_graph()
    state: State = {
        "messages": [],
    }

    while True:
        user_input = input("User: ")
        if user_input.lower() in {"exit", "quit", "bye"}:
            print("Goodbye!")
            break

        state["messages"] = state.get("messages", []) + [
            {"role": "user", "content": user_input}
        ]
        state = graph.invoke(state, {"recursion_limit": 100})

        if state.get("messages") and len(state["messages"]) > 0:
            reply = state["messages"][-1]
            print(f"Assistant: \n\n{reply.content}\n")


if __name__ == "__main__":
    run_chatbot()
