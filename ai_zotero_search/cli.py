import os
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
import click


DEFAULT_ZOTERO_DB = os.getenv("DEFAULT_ZOTERO_DB", os.path.expanduser("~/.zotero/Zotero/zotero.sqlite"))


def setup_zotero_db(database_uri: str = DEFAULT_ZOTERO_DB):
    db = SQLDatabase.from_uri(database_uri)
    return db


def setup_zotero_agent(db: SQLDatabase):
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    agent_executor = create_sql_agent(
        llm, db=db, agent_type="openai-tools", verbose=True)
    return agent_executor


@click.group()
def cli():
    pass


@cli.command()
@click.argument(
    "query",
    str,
    default="Search for any entries in the database that reference 'diabetes'."
)
def search(query: str):
    db = setup_zotero_db()
    agent_executor = setup_zotero_agent(db)
    output = agent_executor.invoke(query)
    click.echo(click.style(output, fg="green"), color=True)


if __name__ == "__main__":
    cli()
