from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, List


class JobAdSchema(BaseModel):
    # Basic Information
    job_title: str = Field(..., description="The title of the job position")
    company_name: Optional[str] = Field(default=None, description="The company offering the job")
    job_category: Optional[str] = Field(default=None, description="The general category or industry of the job")
    sub_category: Optional[str] = Field(default=None, description="The more specific sub-category within the job category")
    industry: Optional[str] = Field(default=None, description="The industry or sector of the job")

    # Location
    location: Optional[str] = Field(default=None, description="The specific location of the job (e.g., city, province, or region)")
    country: Optional[str] = Field(default=None, description="The country where the job is based")

    # Posting details
    posting_date: Optional[str] = Field(default=None, description="The date when the job was posted")
    deadline: Optional[str] = Field(default=None, description="The application deadline, if available")
    url: HttpUrl = Field(..., description="The URL of the job advertisement")

    # Job Requirements
    qualifications: Optional[List[str]] = Field(default=None, description="Required qualifications such as degree, major, certifications")
    technical_skills: Optional[List[str]] = Field(default=None, description="Required technical skills (e.g., programming languages, tools)")
    soft_skills: Optional[List[str]] = Field(default=None, description="Required soft skills (e.g., communication, teamwork)")
    languages_required: Optional[List[str]] = Field(default=None, description="Languages required (e.g., Vietnamese, English)")
    experience_required: Optional[float] = Field(default=None, description="Number of years of experience required")

    # Job Conditions
    salary: Optional[str] = Field(default=None, description="The salary or compensation details (e.g., range, negotiable)")
    contract_type: Optional[str] = Field(default=None, description="Type of contract (full-time, part-time, internship, freelance)")
    working_hours: Optional[str] = Field(default=None, description="Working hours (e.g., 9-5, shift-based)")
    benefits: Optional[List[str]] = Field(default=None, description="Benefits offered (insurance, bonuses, training, etc.)")

    # Textual Information (for NLP tasks)
    description: str = Field(..., description="Full job description, including responsibilities")
    requirements_text: Optional[str] = Field(default=None, description="Unstructured requirements section text, for NLP parsing")