# 💬 BOB AI Multilingual Product Assistant

A voice-first Streamlit dashboard that lets customers ask about BOB banking products in their own language and hear the answer back.

**Speak or type → Understand → Match products → Answer → Speak**

## Features

- **Voice or text input** in English, Hindi, Marathi, Tamil, Bengali, Telugu, Kannada, Malayalam and Gujarati
- **Recorder with checks**: blank or too-short recordings are rejected, and you confirm the playback before it is processed
- **Automatic understanding**: language detection, transcription, English translation, product category and keywords (Gemini)
- **Product matching**: catalogue rows are ranked by keyword hits, with the best matches shown in a table you can download as CSV
- **Answer in two languages**: English and the customer's language, with a spoken version (gTTS) you can download as MP3
- **Live progress**: streamed model output and per-stage timings
- **KPI strip, session history, insights charts and 👍/👎 feedback**
- **Secure by default**: the Gemini API key is read from Streamlit secrets, never from source code
- **Developer data downloader** (password protected): scrape the latest product data from the bank website, then download each file or a ZIP

## Project structure

```
.
├── speech.py                      # the Streamlit app
├── requirements.txt
├── README.md
├── img - Copy (2).png             # BOB banner (shown at the top of the app)
├── Bank_Product/                  # product catalogue, one Excel file per category
│   ├── home_loan.xlsx
│   ├── vehicle_loan.xlsx
│   ├── education_loan.xlsx
│   ├── personal_loan.xlsx
│   ├── mortgage_loan.xlsx
│   ├── term_deposits.xlsx
│   ├── savings_account.xlsx
│   ├── salary_account.xlsx
│   ├── current_account.xlsx
│   └── debit_card.xlsx
└── .streamlit/
    └── secrets.toml.example       # template - copy to secrets.toml
```

Each Excel file can have any columns. The matching is done across every cell, so include the product name, features, eligibility and any URL you want the assistant to share.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit .streamlit/secrets.toml and paste your Gemini API key

streamlit run speech.py
```

Get a key from [Google AI Studio](https://aistudio.google.com/apikey).

## Configure the API key (secrets)

The app looks for `GEMINI_API_KEY` in this order:

1. Streamlit secrets (`.streamlit/secrets.toml` locally, or **App settings → Secrets** on Streamlit Cloud)
2. The `GEMINI_API_KEY` environment variable

```toml
GEMINI_API_KEY = "your-gemini-api-key"
```

`.streamlit/secrets.toml` is listed in `.gitignore`. Only the `.example` file is committed.

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository (include `Bank_Product/` and the banner image).
2. On [share.streamlit.io](https://share.streamlit.io), choose **New app**, select the repo and set the main file to `speech.py`.
3. Open **Advanced settings → Secrets** and paste:
   ```toml
   GEMINI_API_KEY = "your-gemini-api-key"
   ```
4. Deploy. The microphone works because Streamlit Cloud serves the app over HTTPS.

## Settings in the sidebar

| Setting | What it does |
|---|---|
| Voice Model | Gemini model that understands the audio or text |
| Generation Model | Model that writes the final answer |
| Auto-play spoken answer | Plays the answer as soon as it is ready |
| Products to consider | How many matched products are passed to the model |

## Developer options (refreshing product data)

The sidebar has a **🛠️ Developer Options** section. Enter the developer password (default `department`; set `DEV_PASSWORD` in secrets to change it), then press **📥 Open Data Downloader**.

1. Pick one or more categories in the multiselect (or **Select all**).
2. Optionally open **Source pages** to check or edit the listing URL for each category.
3. Press **Start download**. Progress is shown for every category and every product page.
4. Download each Excel file from its own row, or take everything with **Download all (ZIP)**.
5. Edit the files if you wish, put them in `Bank_Product/` with the same names, and commit.

Notes:
- The scraper reads the product links on a category page (`<li class="item">`) and then each product page, using the bank team's scrape logic. Pages it cannot read are skipped and listed with the reason, so you can retry.
- Category pages are built from the page names (for example `saving-accounts` becomes `https://bankofbaroda.bank.in/accounts/saving-accounts`). You can edit any URL under **Source pages**.
- The menu has 12 items. The first 10 match the files the assistant reads. **Recurring Deposit** and **Fixed Deposit** are extra downloads.
- Files written on Streamlit Cloud are not permanent, which is why the downloader hands you the files instead of saving them on the server.

## Troubleshooting

| Problem | Fix |
|---|---|
| "Gemini API key not found" | Add `GEMINI_API_KEY` to secrets as shown above |
| Microphone does not record | Allow mic access in the browser; use `localhost` or HTTPS |
| "Product data … was not found" | Check the file name in `Bank_Product/` matches the category, with underscores (e.g. `home_loan.xlsx`) |
| Banner missing | Make sure `img - Copy (2).png` is in the repo root; a coloured text banner is shown as a fallback |
| No spoken answer | gTTS needs internet access; the written answer is still shown |
| Downloader finds no product pages | Open **Source pages**, check the URL in a browser and make sure the page lists products as links |
| Model dropdown shows only defaults | The model list could not be fetched; check the key and quota |

## Disclaimer

Answers are AI-generated from the product catalogue and are for guidance only. Confirm rates, eligibility and charges with the bank before applying.
