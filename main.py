import sys
import time
import pytz
from datetime import datetime

from utils import get_daily_papers_by_keyword_with_retries, generate_table, back_up_files,\
    restore_files, remove_backups, get_daily_date


beijing_timezone = pytz.timezone('Asia/Singapore')

# NOTE: arXiv API seems to sometimes return an unexpected empty list.

# get current beijing time date in the format of "2021-08-01"
current_date = datetime.now(beijing_timezone).strftime("%Y-%m-%d")
# get last update date from README.md
with open("README.md", "r") as f:
    while True:
        line = f.readline()
        if "Last update:" in line: break
    last_update_date = line.split(": ")[1].strip()
    if last_update_date == current_date:
        sys.exit("Already updated today!")

keywords = ["Time Series", "LLM", "Image Restoration", 
            "Diffusion Model", "Multimodal",
            "Photoacoustic Image"] # TODO add more keywords

max_result = 100 # maximum query results from arXiv API for each keyword
issues_result = 20 # maximum papers to be included in the issue

# all columns: Title, Authors, Abstract, Link, Tags, Comment, Date
# fixed_columns = ["Title", "Link", "Date"]

column_names = ["Title", "Link", "Cool Paper", "Abstract", "Date", "Comment"]

back_up_files() # back up README.md and ISSUE_TEMPLATE.md

# write to README.md
f_rm = open("README.md", "w") # file for README.md
f_rm.write("# Daily Papers\n")
f_rm.write("The project automatically fetches the latest papers from arXiv based on keywords.\n\nThe subheadings in the README file represent the search keywords.\n\nOnly the most recent articles for each keyword are retained, up to a maximum of 100 papers.\n\n")
f_rm.write("You can click the 'Watch' button to receive daily email notifications.\n\n")
f_rm.write("Or you can fork this repository and set your own key words in `main.py`:\n\n")
f_rm.write("Last update: {0}\n\n".format(current_date))
f_rm.write("👍Thanks to [zezhishao/DailyArXiv](https://github.com/zezhishao/DailyArXiv) and [Cool Paper](https://papers.cool).\n\n")

f_rm.write("## Index\n\n")
for keyword in keywords:
    f_rm.write("- [{0}](#{1})\n".format(keyword, keyword.replace(" ", "-")))

# write to ISSUE_TEMPLATE.md
f_is = open(".github/ISSUE_TEMPLATE.md", "w") # file for ISSUE_TEMPLATE.md
f_is.write("---\n")
f_is.write("title: Latest {0} Papers - {1}\n".format(issues_result, get_daily_date()))
f_is.write("labels: documentation\n")
f_is.write("---\n")

for keyword in keywords:
    f_rm.write("## {0}\n".format(keyword))
    f_is.write("## {0}\n".format(keyword))
    f_rm.write("[Back to Index](#Index)\n\n")
    if len(keyword.split()) == 1: link = "AND" # for keyword with only one word, We search for papers containing this keyword in both the title and abstract.
    else: link = "OR"
    papers = get_daily_papers_by_keyword_with_retries(keyword, column_names, max_result, link)
    if papers is None: # failed to get papers
        print("Failed to get papers!")
        f_rm.close()
        f_is.close()
        restore_files()
        sys.exit("Failed to get papers!")
    rm_table = generate_table(papers)
    is_table = generate_table(papers[:issues_result], ignore_keys=["Abstract"])
    f_rm.write(rm_table)
    f_rm.write("\n\n")
    f_is.write(is_table)
    f_is.write("\n\n")
    time.sleep(5) # avoid being blocked by arXiv API

f_rm.close()
f_is.close()
remove_backups()
