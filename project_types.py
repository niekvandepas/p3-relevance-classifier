from typing import TypedDict


class RedditItem(TypedDict):
    id: str
    text: str


class DelpherItem(TypedDict):
    publication_date: str
    title: str
    ocr_url: str
    paper_title: str
    spatial_creation: str
    identifier: str
    ocr_xml: str
    plain_text: str
