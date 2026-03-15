import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tqdm import tqdm

BASE_URL = "https://www.uhcprovider.com"
PAGE_URL = "https://www.uhcprovider.com/en/policies-protocols/commercial-policies/commercial-medical-drug-policies.html"

SAVE_DIR = "data/pdfs"
os.makedirs(SAVE_DIR, exist_ok=True)


def get_pdf_links():

    print("Fetching webpage...")

    response = requests.get(PAGE_URL)
    soup = BeautifulSoup(response.text, "html.parser")

    pdf_links = []

    for link in soup.find_all("a", href=True):

        href = link["href"]

        if ".pdf" in href.lower():
            full_link = urljoin(BASE_URL, href)
            name = link.text.strip()

            pdf_links.append({
                "name": name,
                "url": full_link
            })

    print(f"Found {len(pdf_links)} PDF policies")

    return pdf_links


def download_pdfs(pdf_links):

    for pdf in tqdm(pdf_links):

        filename = pdf["url"].split("/")[-1]
        path = os.path.join(SAVE_DIR, filename)

        if os.path.exists(path):
            continue

        try:
            r = requests.get(pdf["url"], timeout=60)

            with open(path, "wb") as f:
                f.write(r.content)

        except Exception as e:
            print("Failed:", pdf["url"], e)


if __name__ == "__main__":

    links = get_pdf_links()

    download_pdfs(links)

    print("Download complete")