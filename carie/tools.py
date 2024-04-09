import json

from dspy.predict.parameter import Parameter
from dspy.primitives.prediction import Prediction
from duckduckgo_search import DDGS
from serpapi import Client as SerpApiClient

from config import SERP_API_KEY
from plants import get_plant, list_plants


class RetrieverTool(Parameter):
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def reset(self):
        pass

    def dump_state(self):
        return {}

    def load_state(self, state):
        for name, value in state.items():
            setattr(self, name, value)


class ExaminePlant(RetrieverTool):
    name = "examine_plant"
    input_variable = "plant_name"
    desc = "selects one of our plants by its name and describes their overall characteristics"

    def forward(self, plant_name: str, *args, **kwargs) -> Prediction:
        plant = get_plant(plant_name)

        if not plant:
            return Prediction(passages=[f"We dont have a plant named `{plant_name}`"])

        exam = f"{plant_name} is a {plant.scientific_name}. It currently is {plant.status}."
        passages = [exam]
        prediction = Prediction(passages=passages)

        return prediction


class ReadPlantSensor(RetrieverTool):
    name = "read_plant_sensor"
    input_variable = "plant_name, sensor_name"
    desc = "selects one of our plants by its name and read one of its sensors. Available sensors: air_humidity, air_temperature, soil_humidity, soil_ph, light_level"

    def forward(self, input_variable: str, *args, **kwargs) -> Prediction:
        try:
            plant_name, sensor_name = [
                variable.strip() for variable in input_variable.split(",")
            ]
        except ValueError:
            return Prediction(
                passages=[
                    "The action MUST follow the format read_plant_sensor[one of our plants name, one of the available sensors]"
                ]
            )

        if sensor_name not in [
            "air_humidity",
            "air_temperature",
            "soil_humidity",
            "soil_ph",
            "light_level",
        ]:
            return Prediction(passages=[f"We dont have a sensor named `{sensor_name}`"])

        plant = get_plant(plant_name)

        if not plant:
            return Prediction(passages=[f"We dont have a plant named `{plant_name}`"])

        clean_sensor_name = sensor_name.replace("_", " ")
        actual_sensor_value = plant.actual_sensor.__getattribute__(sensor_name)
        ideal_max_sensor_value = plant.ideal_max_sensor.__getattribute__(sensor_name)
        ideal_min_sensor_value = plant.ideal_min_sensor.__getattribute__(sensor_name)

        passages = [
            f"{plant.name}'s {clean_sensor_name} currently is {actual_sensor_value}. Ideally it should be between {ideal_min_sensor_value} and {ideal_max_sensor_value}"
        ]
        prediction = Prediction(passages=passages)

        return prediction


class ListPlants(RetrieverTool):
    name = "list_plants"
    input_variable = " "
    desc = "lists all of our plant's names and species"

    def forward(self, *args, **kwargs) -> Prediction:
        plants = list_plants()

        passages = [f"{plant.name}, currently has {plant.status}" for plant in plants]

        prediction = Prediction(passages=passages)
        return prediction


class WebSearch(RetrieverTool):
    name = "web_search"
    input_variable = "query"
    desc = "takes a search query and returns one or more potentially relevant results from a web search engine"

    def forward(self, query: str, max_results: int = 5, *args, **kwargs) -> Prediction:
        clean_query = query.split("]")[0]

        passages = []
        try:
            passages = list(self._duckduckgo(clean_query, max_results))
        except Exception:
            passages = list(self._google(clean_query, max_results))

        prediction = Prediction(passages=passages)
        return prediction

    @staticmethod
    def _duckduckgo(query: str, max_results: int):
        with DDGS() as ddgs:
            for result in ddgs.text(query, max_results=max_results):
                yield json.dumps(
                    {
                        "title": result["title"],
                        "content": result["body"],
                    }
                )

    @staticmethod
    def _google(query: str, max_results: int):
        client = SerpApiClient(api_key=SERP_API_KEY)
        results = client.search(q=query, engine="google")

        for organic_result in results["organic_results"][:max_results]:
            yield json.dumps(
                {
                    "title": organic_result["title"],
                    "content": organic_result["snippet"],
                }
            )
