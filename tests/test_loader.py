from src.loader import parse_meter_column


def test_normal_meter_is_consumption():
    meter_id, flow_type = parse_meter_column("DFE2001117")

    assert meter_id == "DFE2001117"
    assert flow_type == "consumption"


def test_prosumer_a_plus_is_consumption():
    meter_id, flow_type = parse_meter_column("DFE9013260 - A+")

    assert meter_id == "DFE9013260"
    assert flow_type == "consumption"


def test_prosumer_a_minus_is_injection():
    meter_id, flow_type = parse_meter_column("DFE9013260 - A-")

    assert meter_id == "DFE9013260"
    assert flow_type == "solar_injection"
