import hydra
from discord.ext.commands import Bot

from classes import CashMannager, Predictor

bot = Bot(command_prefix="")
cash = None
predictor = None


@bot.event
async def on_ready():
    print(f"logged in as {bot.user}")


@bot.event
async def on_message(message):
    if message.author.bot:  # 봇 무시 (본인 포함)
        return

    cash.add_message(message)

    if message.content == "테스트":
        await message.channel.send(cash.get_messages(message))


@hydra.main(config_name="config.yaml")
def main(cfg):
    global cash, predictor
    predictor = Predictor(cfg.DISCORD.tokenizer, cfg.DISCORD.model)
    cash = CashMannager(predictor.tokenizer)

    bot.run(cfg.DISCORD.token)


if __name__ == "__main__":
    main()
