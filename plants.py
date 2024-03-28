from dataclasses import dataclass
import json

_PLANTS = []
_PLANTS_JSON_FILE_PATH = "./storage/plants.json"


@dataclass
class Sensor:
    air_humidity: float  # Typical Range: 0% (completely dry) to 100% (completely saturated with moisture).
    air_temperature: float  # Typical Range: -40°C to 50°C (-40°F to 122°F).
    soil_humidity: float  # Typical Range: 0% (completely dry) to 100% (saturated soil).
    soil_ph: float  # Typical Range: 0 (highly acidic) to 14 (highly alkaline).
    light_level: int


@dataclass
class Plant:
    id: int
    name: str
    personality: str
    scientific_name: str
    actual_sensor: Sensor
    ideal_min_sensor: Sensor
    ideal_max_sensor: Sensor

    @property
    def genus(self):
        return self.scientific_name.split(" ")[0]

    @property
    def status(self) -> str:
        status = []

        for attribute in [
            "air_humidity",
            "air_temperature",
            "soil_humidity",
            "soil_ph",
            "light_level",
        ]:
            actual = self.actual_sensor.__getattribute__(attribute)
            ideal_min = self.ideal_min_sensor.__getattribute__(attribute)
            ideal_max = self.ideal_max_sensor.__getattribute__(attribute)

            readable_attribute = attribute.replace("_", " ")

            if actual < ideal_min:
                status.append(f"low {readable_attribute}")
            elif actual > ideal_max:
                status.append(f"high {readable_attribute}")
            else:
                status.append(f"good {readable_attribute}")

        status = ", ".join(status)
        return status

    @property
    def summary(self):
        summary = {
            "name": self.name,
            "scientific_name": self.scientific_name,
            "status": self.status,
        }

        return summary

    @property
    def exam(self):
        exam = {
            "name": self.name,
            "scientific_name": self.scientific_name,
            "personality": self.personality,
        }

        for attribute in [
            "air_humidity",
            "air_temperature",
            "soil_humidity",
            "soil_ph",
            "light_level",
        ]:
            exam[attribute] = {
                "ideal_min": self.ideal_min_sensor.__getattribute__(attribute),
                "actual": self.actual_sensor.__getattribute__(attribute),
                "ideal_max": self.ideal_max_sensor.__getattribute__(attribute),
            }

        return exam


def list_plants() -> list[Plant]:
    """Lists available plants.

    Returns:
        A list with available Plant instances.
    """
    global _PLANTS

    if _PLANTS:
        return _PLANTS

    with open(_PLANTS_JSON_FILE_PATH) as file:
        plants_json = json.load(file)

    for plant_json in plants_json:
        del plant_json["_meta"]
        plant = Plant(**plant_json)
        plant.actual_sensor = Sensor(**plant_json["actual_sensor"])
        plant.ideal_min_sensor = Sensor(**plant_json["ideal_min_sensor"])
        plant.ideal_max_sensor = Sensor(**plant_json["ideal_max_sensor"])
        _PLANTS.append(plant)

    return _PLANTS


def get_plant(name: str) -> Plant | None:
    """Gets a Plant by its name.

    Args:
        name: the name of the Plant to be detailed.

    Returns:
        Plant matching `name` or None if there is no match.
    """
    if not name:
        return None

    lower_name = name.lower()

    plants = list_plants()
    plant = next((plant for plant in plants if plant.name.lower() == lower_name), None)

    return plant
