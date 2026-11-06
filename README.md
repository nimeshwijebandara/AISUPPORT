# AI-Based IT Helpdesk and Automated Troubleshooting System

An MSc Computer Science project that combines an IT helpdesk ticket workflow with AI-assisted issue classification and knowledge retrieval.

## Project overview

The application provides:

- Secure, role-based login for administrators, support staff and end users.
- AI-assisted IT issue classification.
- Knowledge-base retrieval for troubleshooting guidance.
- Support ticket creation and ticket status tracking.
- Ticket conversations, comments and file attachments.
- Human review of low-confidence AI predictions.
- SLA monitoring and IT support performance analytics.
- Email notifications, where SMTP settings are configured.
- SQLite for local development and PostgreSQL for hosted deployment.

## Technology stack

- Python
- Streamlit
- PostgreSQL / SQLite
- pandas
- scikit-learn model loaded with joblib
- Sentence Transformers
- ChromaDB
- bcrypt

## Project structure

```text
AI_IT_HELPDESK/
├── backend/
│   └── database.py
├── frontend/
│   └── app.py
├── database/
│   └── helpdesk.db              # Local SQLite database, if used
├── models/
│   └── it_helpdesk_classifier_v4.pkl
├── knowledge_base/
│   ├── it_knowledge_base.csv
│   └── vector_store/            # Created/used by ChromaDB
├── uploads/
│   └── ticket_attachments/
├── requirements.txt
├── .env                         # Local secrets; do not commit
└── README.md
```

Some directories are created by the application when it runs. Keep required model and knowledge-base files in the paths expected by `frontend/app.py`.

## Prerequisites

- Python version compatible with the packages in `requirements.txt`.
- Git.
- Access to the configured database.
- The model file and knowledge-base CSV used by the application.

## Installation

Open Command Prompt in the project directory:

```cmd
cd /d D:\AI_IT_HELPDESK
```

Create and activate a virtual environment (optional but recommended):

```cmd
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```cmd
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For PostgreSQL deployments, ensure `requirements.txt` includes the PostgreSQL driver used by `backend/database.py`, for example:

```text
psycopg2-binary>=2.9.10
```

Do not install both `psycopg2` and `psycopg2-binary` unnecessarily in the same environment.

## Configuration

### Local environment

Configure the database settings using the variable names and format expected by `backend/database.py`.

Example structure (replace placeholders locally; never commit real credentials):

```dotenv
DATABASE_MODE=postgres
DATABASE_URL=postgresql://DB_USER:DB_PASSWORD@DB_HOST:5432/DB_NAME
```

For local SQLite development, use the mode supported by your `backend/database.py` configuration.

If email notifications are required, configure the SMTP variables read by `frontend/app.py`:

```dotenv
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=admin@email.com
SMTP_PASSWORD=app-password
EMAIL_FROM=admin@email.com
```

Use an email-provider app password where required. Do not place real passwords or tokens in this README, source code, screenshots, or Git commits.

### Streamlit Community Cloud

1. Push the project to the GitHub repository connected to Streamlit.
2. Set the application entry point to `frontend/app.py`.
3. Add the required configuration values in **App settings → Secrets**, using the format supported by `backend/database.py`.
4. Ensure the database driver is listed in `requirements.txt`.
5. Reboot the application after changing dependencies or secrets.
6. Review **Manage app → Logs** if deployment or runtime errors occur.

Never publish database URLs containing passwords. If a credential has been exposed, rotate it at the provider and update the deployment secret.

## Run locally

From the project root:

```cmd
streamlit run frontend/app.py
```

If `streamlit` is not recognised:

```cmd
python -m streamlit run frontend/app.py
```

## Database compatibility note

Database access should go through the shared adapter in `backend/database.py`, rather than opening a database-specific connection directly in individual UI sections.

For example, use the adapter pattern:

```python
import database

with database.database_connection() as conn:
    rows = database.fetch_all(
        conn,
        "SELECT user_id, username FROM users"
    )
```

The performance analytics section should also use this adapter. Avoid using `sqlite3.connect(DB_PATH)` or passing a raw PostgreSQL connection to `pandas.read_sql_query()` in a section that must work with both database modes.

## Main application functions

### AI-assisted issue analysis

The user describes an IT issue. The application predicts an issue category and retrieves related knowledge-base documents. Low-confidence predictions can be routed for human review according to the configured confidence threshold.

### Ticket management

Users can create and view tickets. Support staff and administrators can manage tickets according to their roles. Ticket conversations and permitted attachments are associated with tickets.

### Human review

Administrators can review predictions routed for human review, confirm or correct the category, and export reviewed feedback records.

### SLA and performance analytics

The dashboard presents ticket status, response and resolution information, SLA monitoring and support-staff workload. Analytics should be tested against the configured database mode.

## Verification checklist

After deployment, test the following with authorised test accounts:

- [ ] Application starts without import or dependency errors.
- [ ] Existing administrator can log in.
- [ ] Existing end user can log in.
- [ ] Existing support user can log in.
- [ ] Role-based navigation and access restrictions work.
- [ ] AI issue classification returns a category.
- [ ] Knowledge retrieval returns relevant records when matches exist.
- [ ] Ticket creation persists to the configured database.
- [ ] Ticket comments persist and display correctly.
- [ ] Support assignment and status updates persist.
- [ ] Attachment upload and download work.
- [ ] Human-review queue and feedback export work.
- [ ] SLA dashboard loads.
- [ ] Performance analytics loads without a database error.
- [ ] Email notifications are tested only when SMTP is configured.
- [ ] Cloud deployment logs contain no unhandled application errors.

Record actual test outcomes, dates and screenshots. Do not report a test as passed until it has been executed successfully.

## Troubleshooting

### `ModuleNotFoundError: No module named 'psycopg2'`

Add the PostgreSQL driver used by the database adapter to `requirements.txt`, commit and push the change, then allow Streamlit Cloud to reinstall dependencies.

### Cloud login fails

Check the application logs for the underlying exception. Confirm the database mode and connection configuration, verify that the expected user record exists, and confirm the stored password hash is compatible with the authentication code. Do not expose passwords or hashes in logs.

### Analytics page reports a database error

Check that the analytics section uses `database.database_connection()` and `database.fetch_all()` from the shared adapter. Confirm that the SQL column and table names match the deployed database schema. Inspect the full server-side traceback; the browser's shortened traceback may not show the root cause.

### `NameError: __file__ is not defined` in Jupyter

`__file__` is normally defined when Python executes a script file. It is generally not defined when code is run directly in a Jupyter notebook cell. Run the Streamlit application as a script, or use an explicit project path for notebook-only tests.

## Security

- Keep `.env`, passwords, API keys and database URLs out of Git.
- Use Streamlit Secrets for hosted credentials.
- Rotate credentials that have been exposed.
- Use parameterised SQL queries.
- Do not log passwords, password hashes, tokens or full connection strings.
- Restrict administrator and support functions by role.
- Validate uploaded file types and sizes.
- Use test accounts and non-sensitive test data for demonstrations.

## Dissertation evidence

Capture genuine evidence from the running application for:

- Cloud login success.
- Role-based dashboards.
- Ticket creation, assignment, comments and status updates.
- AI classification and confidence display.
- Knowledge retrieval.
- Human review and feedback export.
- SLA and performance analytics.
- Database persistence and relevant test results.

For model and RAG evaluation, report the actual results produced by the evaluation scripts/CSV files. Do not invent metrics or present development tests as independent evaluation results.

## Repository maintenance

Before committing:

```cmd
git status
```

Check that `.env`, credentials, local database files containing personal data, uploaded attachments and generated secrets are not staged.

Example commit and push:

```cmd
git add frontend/app.py README.md requirements.txt
git commit -m "Fix PostgreSQL analytics and update project documentation"
git push origin main
```

## Project status

This README describes the current application structure and intended test areas. Features and test outcomes should be marked complete only after they have been verified in the target environment.
