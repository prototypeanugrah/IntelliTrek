# agents/trail_agent.py
import ast
import sys
from pprint import pprint
from typing import List

sys.path.append("..")

import pandas as pd
from pydantic import BaseModel

import utils


class SheetsResponse(BaseModel):
    name: str
    area_name: str
    city_name: str
    state_name: str
    lat: float
    lng: float
    popularity: float
    length: float
    elevation_gain: float
    difficulty_rating: int
    route_type: str
    visitor_usage: int
    avg_rating: float
    num_reviews: int
    features: List[str]
    activities: List[str]
    units: str


class Sheet(BaseModel):
    sheet_df: List[SheetsResponse]


class SheetsAgent:
    def __init__(
        self,
        credentials_path: str = "credentials.json",
        sheet_id: str = "1R17IankLGDqamjiAUvBnhxYockM3_y7TG56BpduH8gc",
    ):
        self.client = utils.create_google_client(credentials_path)
        self.sheet_id = sheet_id

    def get_sheet(self) -> Sheet:
        ws = self.client.open_by_key(self.sheet_id).sheet1
        df = pd.DataFrame(ws.get_all_records())
        df["_geoloc"] = df["_geoloc"].apply(ast.literal_eval)
        df["lat"] = df["_geoloc"].apply(lambda x: x.get("lat"))
        df["lng"] = df["_geoloc"].apply(lambda x: x.get("lng"))
        df["lat"] = df["lat"].astype(float)
        df["lng"] = df["lng"].astype(float)
        df.drop(
            columns=[
                "trail_id",
                "country_name",
                "_geoloc",
            ],
            inplace=True,
        )
        df["visitor_usage"] = df["visitor_usage"].apply(lambda x: int(x) if x else 0)
        df["features"] = df["features"].apply(ast.literal_eval)
        df["activities"] = df["activities"].apply(ast.literal_eval)

        # Remove the entries which contain 'CLOSED' in the name
        df = df[~df["name"].str.contains("CLOSED")]

        return Sheet(
            sheet_df=[SheetsResponse(**trail) for trail in df.to_dict("records")]
        )


if __name__ == "__main__":
    sheets_agent = SheetsAgent()

    sheet = sheets_agent.get_sheet()

    # print the first 5 trails
    for trail in sheet.sheet_df[:2]:
        pprint(trail.model_dump())
        print("\n")
