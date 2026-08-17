from pathlib import Path
import re

import pandas as pd


BASE_COLUMNS = [
    "PeriodYear",
    "PeriodMonth",
    "WeekDay",
    "Date",
    "Hour",
    "Tariff",
]

PROSUMER_PATTERN = re.compile(
    r"^(?P<meter_id>.+?)\s*-\s*(?P<flow_type>A[+-])$"
)


def parse_meter_column(column_name: str) -> tuple[str, str]:
    column_name = str(column_name).strip()
    match = PROSUMER_PATTERN.match(column_name)

    if not match:
        return column_name, "consumption"

    meter_id = match.group("meter_id").strip()
    flow_type = match.group("flow_type")

    if flow_type == "A+":
        return meter_id, "consumption"

    return meter_id, "solar_injection"


def load_sheet(
    excel_file: pd.ExcelFile,
    sheet_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_excel(
        excel_file,
        sheet_name=sheet_name,
        header=[0, 1],
    )

    base_df = df.iloc[:, :6].copy()
    base_df.columns = BASE_COLUMNS

    meter_df = df.iloc[:, 6:].copy()
    meter_df.columns = [
        str(column[1]).strip()
        for column in meter_df.columns
    ]

    metadata = []

    for company_code, column_name in df.columns[6:]:
        meter_id, flow_type = parse_meter_column(column_name)

        metadata.append(
            {
                "source_sheet": sheet_name,
                "company_code": str(company_code).strip(),
                "source_column": str(column_name).strip(),
                "meter_id": meter_id,
                "flow_type": flow_type,
            }
        )

    metadata_df = pd.DataFrame(metadata)

    return base_df, meter_df, metadata_df


def get_sheet_names(path: str | Path) -> list[str]:
    with pd.ExcelFile(path, engine="openpyxl") as excel_file:
        return excel_file.sheet_names