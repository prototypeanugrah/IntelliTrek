import asyncio
import os
from pprint import pprint

import aiohttp
import geocoder
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


class Weather(BaseModel):
    city: str
    state: str
    lat: float
    lon: float
    temperature: float
    feels_like: float
    condition: str
    condition_description: str
    humidity: float
    wind: float


class WeatherAgent:
    BASE = "https://api.openweathermap.org/data/2.5/weather"

    def __init__(
        self,
        city: str,
        state: str,
        openweather_api_key: str,
        google_api_key: str = os.getenv("GOOGLE_API_KEY"),
        session: aiohttp.ClientSession | None = None,
    ):
        self.city = city
        self.state = state
        self.openweather_api_key = openweather_api_key
        self.google_api_key = google_api_key
        self.session = session  # Do not create a session here

    async def get_lat_lon(self, location: str) -> tuple[float, float]:
        g = geocoder.google(location, key=self.google_api_key)
        if not g.latlng:
            raise ValueError(f"Could not geocode location: {location}")
        return g.latlng[0], g.latlng[1]

    async def get_weather(self, lat: float, lon: float) -> Weather:
        params = {
            "lat": lat,
            "lon": lon,
            "units": "imperial",
            "appid": self.openweather_api_key,
        }
        if self.session is not None:
            async with self.session.get(self.BASE, params=params) as r:
                if r.status != 200:
                    raise RuntimeError(
                        f"OpenWeather error {r.status}: {await r.text()}"
                    )
                data = await r.json()
        else:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.BASE, params=params) as r:
                    if r.status != 200:
                        raise RuntimeError(
                            f"OpenWeather error {r.status}: {await r.text()}"
                        )
                    data = await r.json()
        temp = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        cond = data["weather"][0]["main"].lower()
        cond_desc = data["weather"][0]["description"]
        humidity = data["main"]["humidity"]
        wind = data["wind"]["speed"]  # mph because units=imperial
        ltd = data["coord"]["lat"]
        lng = data["coord"]["lon"]
        return Weather(
            city=self.city,
            state=self.state,
            lat=ltd,
            lon=lng,
            temperature=temp,
            feels_like=feels_like,
            condition=cond,
            condition_description=cond_desc,
            humidity=humidity,
            wind=wind,
        )


async def main():
    input_location = input("Enter a location in this format: City, State: ")
    city, state = input_location.split(",")
    weather_agent = WeatherAgent(
        city=city,
        state=state,
        openweather_api_key=os.getenv("OPENWEATHER_API_KEY"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        session=aiohttp.ClientSession(),
    )
    g = geocoder.google(
        f"{weather_agent.city}, {weather_agent.state}", key=weather_agent.google_api_key
    )
    if not g.latlng:
        print(
            f"Could not geocode location: {input_location}. Please check the input format or your API key."
        )
    else:
        print(f"Latitude: {g.latlng[0]}, Longitude: {g.latlng[1]}")
        result = await weather_agent.get_weather(g.latlng[0], g.latlng[1])
        pprint(result.model_dump())
        print(
            f"City: {result.city}, {result.state}\n"
            f"Condition: {result.condition.capitalize()}\n"
            f"Temperature: {result.temperature:.2f} °F (feels like {result.feels_like:.2f} °F)\n"
            f"Humidity: {result.humidity}%\n"
            f"Wind: {result.wind:.2f} mph"
        )
    await weather_agent.session.close()


if __name__ == "__main__":
    asyncio.run(main())
