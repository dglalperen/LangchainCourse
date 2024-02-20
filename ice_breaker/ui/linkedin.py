import os
from dotenv import load_dotenv
import requests


def scrape_linkedin_profile(linked_in_profile_url: str):
    """
    Scrape information from LinkedIn profiles, Manually scrape the information from the LinkedIn profile
    """
    api_endpoint = "https://nubela.co/proxycurl/api/v2/linkedin"
    header_dic = {
        "Authorization": f"Bearer {os.getenv('PROXYCURL_API_KEY')}",
        "Content-Type": "application/json",
    }
    response = requests.get(
        api_endpoint, params={"url": linked_in_profile_url}, headers=header_dic
    )

    data = response.json()
    data = {
        k: v
        for k, v in data.items()
        if v not in [[], "", "", None]
        and k not in ["people_also_viewed", "certifications"]
    }

    if data.get("groups"):
        for group_dict in data.get("groups"):
            group_dict.pop("profile_pic_url", None)

    return data