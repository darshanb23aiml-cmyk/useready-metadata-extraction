"""Output schema for the six metadata fields."""
from typing import Optional
from pydantic import BaseModel, Field


class AgreementMetadata(BaseModel):
    agreement_value: Optional[str] = Field(
        None, description="Monthly rent as digits only, e.g. 12000. No commas, no currency symbol.")
    agreement_start_date: Optional[str] = Field(
        None, description="Start date of the agreement in DD.MM.YYYY format.")
    agreement_end_date: Optional[str] = Field(
        None, description="End date of the agreement in DD.MM.YYYY format.")
    renewal_notice_days: Optional[str] = Field(
        None, description="Notice period in days, digits only. null if the document does not state one.")
    party_one: Optional[str] = Field(
        None, description="Owner / landlord / lessor name, without titles or addresses.")
    party_two: Optional[str] = Field(
        None, description="Tenant / lessee name, without titles or addresses.")


# Mapping from schema field -> column name used in the provided CSV files
# (note: the CSVs spell "Aggrement", we keep it so evaluation lines up).
CSV_COLUMNS = {
    "agreement_value": "Aggrement Value",
    "agreement_start_date": "Aggrement Start Date",
    "agreement_end_date": "Aggrement End Date",
    "renewal_notice_days": "Renewal Notice (Days)",
    "party_one": "Party One",
    "party_two": "Party Two",
}
