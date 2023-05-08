import os

import discord

from qa_retrieval import build_qa

intents = discord.Intents(messages=True)
intents.message_content = True

client = discord.Client(intents=intents)

qa = build_qa("../aim/docs/source")


@client.event
async def on_ready():
    print(f"We have logged in as {client.user}")


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith("/question"):
        await message.channel.send("One second, searching the documentation...")

        answer = qa.run(message.content)
        await message.channel.send(answer)


client.run(os.environ["DISCORD_TOKEN"])
