import os
from bs4 import BeautifulSoup
import re

def convert_html_to_text(html_path):
    with open(html_path, encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    output_lines = []

    # titles
    for header in soup.find_all(["h1", "h2", "h3"]):
        output_lines.append(f"\n{header.get_text(strip=True)}\n")

    # paragraphs
    for p in soup.find_all("p"):
        output_lines.append(p.get_text(strip=True))

    # lists
    for ul in soup.find_all("ul"):
        for li in ul.find_all("li"):
            output_lines.append(f"- {li.get_text(strip=True)}")

    # tables
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if not rows:
            continue

        headers = [th.get_text(strip=True) for th in rows[0].find_all("th")]
        for row in rows[1:]:
            cells = row.find_all("td")
            if not cells or len(cells) != len(headers):
                continue
            row_title = cells[0].get_text(strip=True)
            for i in range(1, len(cells)):
                column_title = headers[i]
                html_content = cells[i].decode_contents()
                html_content = re.sub(r"<\s*strong>", "\n<strong>", html_content)
                benefit_lines = html_content.split("\n")
                for line in benefit_lines:
                    line = BeautifulSoup(line, "html.parser").get_text(strip=True)
                    # Split by bold title (זהב, כסף, ארד)
                    match = re.match(r"(זהב|כסף|ארד):\s*(.*)", line)
                    if match:
                        tier = match.group(1)
                        detail = match.group(2)
                        output_lines.append(f"{row_title} / {column_title} / {tier}: {detail}")


    return "\n".join(output_lines)


input_dir = "phase2_data"
output_dir = "phase2_data_txt"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith(".html"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename.replace(".html", ".txt"))

        text = convert_html_to_text(input_path)

        with open(output_path, "w", encoding="utf-8") as f_out:
            f_out.write(text)

print("✅ Conversion completed for all HTML files.")
