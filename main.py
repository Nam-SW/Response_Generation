from discord.ext import commands


bot = commands.Bot(command_prefix='')


class ResponseManager(commands.Cog):
    def __init__(self, bot_temp, tokenizer_name, model_name):
        self.bot = bot_temp

        self.tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    @commands.command(
        hidden=False,
        name="",
        usage="",
    )
    async def response(self, message):
        channel_id = message.channel.id
        

@bot.event
async def on_ready():
    print(f'logged in as {bot.user}')


@hydra.main(config_name="config.yaml")
def main(cfg):
    bot.run(cfg.DISCORD.token)


if __name__ == "__main__":
    main()
