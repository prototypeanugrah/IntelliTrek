import asyncio
import os
from typing import Any, List

import aiohttp
import googlemaps
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


class GoogleMapsResponse(BaseModel):
    origin: str
    destinations: str
    destinations_latlng: str
    distance: str
    duration: str


class GoogleMaps(BaseModel):
    maps_df: List[GoogleMapsResponse]


class GoogleMapsAgent:
    BASE = "https://maps.googleapis.com/maps/api/distancematrix/json?"

    def __init__(
        self,
        google_api_key: str,
        session: aiohttp.ClientSession | None = None,
    ):
        self.google_api_key = google_api_key
        self.session = session  # Do not create a session here

    def get_directions(self, src: List[float], dests: List[Any]) -> GoogleMaps:
        gmaps = googlemaps.Client(key=self.google_api_key)
        src_str = f"{src[0]},{src[1]}"
        # dest_strs = [f"{lat},{lng}" for lat, lng in dests]
        dest_strs = [
            f"{float(latlng.split(',')[0])},{float(latlng.split(',')[1])}"
            for latlng in dests
        ]

        # Make the API call
        response = gmaps.distance_matrix(
            origins=[src_str],
            destinations=dest_strs,
            # mode="driving",  # or "walking", "bicycling", "transit"
            units="imperial",  # or "metric"
        )
        # print(response)

        return GoogleMaps(
            maps_df=[
                GoogleMapsResponse(
                    origin=response["origin_addresses"][0],
                    destinations=response["destination_addresses"][i]
                    if element.get("status") == "OK"
                    else "None",
                    distance=element["distance"]["text"]
                    if element.get("status") == "OK"
                    else "None",
                    duration=element["duration"]["text"]
                    if element.get("status") == "OK"
                    else "None",
                    destinations_latlng=dest_strs[i],
                )
                for i, element in enumerate(response["rows"][0]["elements"])
            ]
        )


async def main():
    google_maps_agent = GoogleMapsAgent(google_api_key=os.getenv("GOOGLE_API_KEY"))
    source = (42.0586, -87.6847)  # Evanston, IL (example)

    destinations = [
        "41.8781,-87.6298",  # Chicago, IL
        "42.0334,-88.0834",  # Schaumburg, IL
        "41.8500,-88.3117",  # Naperville, IL
    ]
    # directions = google_maps_agent.get_directions(
    #     "Buffalo, NY",
    #     [
    #         "Ohio and Erie Canal Towpath: Red Lock to Peninsula",
    #         "Brandywine Gorge Trail",
    #     ],
    # )
    directions = google_maps_agent.get_directions(source, destinations)
    print(directions)


if __name__ == "__main__":
    asyncio.run(main())
