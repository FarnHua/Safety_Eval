import gspread
import csv
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials

def upload_csv_to_gsheets(
    csv_path: str,
    spreadsheet_url: str ="",
    credentials_file: str = "",
    worksheet_name: str = "Results",
    start_cell: str = "A1"
):
    """
    Uploads a local CSV file to a specified worksheet in an existing Google Sheet.

    :param csv_path: Path to the CSV file (e.g. './final_results/my_results.csv').
    :param spreadsheet_url: The link (URL) to your existing Google Sheet.
        Example: "https://docs.google.com/spreadsheets/d/1FRELy8sG...."
    :param credentials_file: Path to your Google service account JSON key file.
    :param worksheet_name: Name of the worksheet/tab to write into. If it does not exist, it will be created.
    :param start_cell: Top-left cell (in A1 notation) from which the CSV data will be inserted.
    """
    # 1) Define the scope required by Google Sheets/Drive
    scope = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]

    # 2) Authenticate with Google using the service account file
    creds = ServiceAccountCredentials.from_json_keyfile_name(credentials_file, scope)
    client = gspread.authorize(creds)

    # 3) Open the existing sheet by URL
    spreadsheet = client.open_by_url(spreadsheet_url)

    # 4) Try to open the specified worksheet; if not present, create it
    try:
        worksheet = spreadsheet.worksheet(worksheet_name)
    except gspread.exceptions.WorksheetNotFound:
        # Create a new worksheet with some default size (e.g., 100 rows, 20 columns)
        worksheet = spreadsheet.add_worksheet(title=worksheet_name, rows="100", cols="20")

    # 5) Read the CSV contents into a list of lists
    df = pd.read_csv(csv_path)
    head = [df.columns.values.tolist()]
    data = head + df.values.tolist()
    
    # 6) Update the worksheet with the CSV content, starting at `start_cell`
    worksheet.update(start_cell, data)

    print(f"✓ Uploaded '{csv_path}' into '{worksheet_name}' of your Google Sheet.")

if __name__ == "__main__":

    ## Upload CSV files to Google Sheets
    ## You can change the paths and URLs as needed.
    ## You also need to have credentials set up for Google Sheets API.
    
    upload_csv_to_gsheets(
        csv_path="../advbench_wildguard_eval/advbench.csv",
        spreadsheet_url="",
        credentials_file=""
        worksheet_name="advbench_WildGuard",
        start_cell="A1"
    )

    upload_csv_to_gsheets(
        csv_path="../hex-phi_wildguard_eval/hex-phi.csv",
        spreadsheet_url="",
        credentials_file="",
        worksheet_name="hex-phi_WildGuard",
        start_cell="A1"
    )
    