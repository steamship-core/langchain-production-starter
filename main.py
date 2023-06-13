import sys

from steamship.utils.repl import AgentREPL

sys.path.insert(0, "src")

if __name__ == "__main__":
    # # when running locally, we can use print statements to capture logs / info.
    # # as a result, we will disable python logging to run. this will keep the output cleaner.
    # with LoggingDisabled():
    #     try:
    #         main()
    #     except SteamshipError as e:
    #         print(colored("Aborting! ", "red"), end="")
    #         print(f"There was an error encountered when running: {e}")
    #

    AgentREPL(LangChainTelegramChatbot, "telegram_respond", agent_package_config={"bot_token": "test"}).run()
