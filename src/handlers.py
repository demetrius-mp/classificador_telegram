from io import BytesIO

from PIL import Image
from telegram import Update
from telegram.ext import CommandHandler, ContextTypes, MessageHandler, filters

from classifier import classifier


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = """Comandos disponíveis:
- /help -> exibe uma mensagem de ajuda.
"""
    await update.message.reply_text(msg)


async def help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"Esta rede neural foi treinada utilizando as seguintes classes: {', '.join(classifier.labels)}."
    )

    await update.message.reply_text("Para classificar uma imagem, basta enviá-la aqui!")


async def classify(update: Update, context: ContextTypes.DEFAULT_TYPE):
    telegram_picture = await context.bot.get_file(update.message.photo[-1].file_id)
    f = BytesIO(await telegram_picture.download_as_bytearray())

    img = Image.open(f)
    predicted_label = classifier(img)

    await update.message.reply_text(f"Isso é um(a) {predicted_label}!")


handle_start = CommandHandler("start", start)
handle_help = CommandHandler("help", help)
handle_classify = MessageHandler(filters.PHOTO & ~filters.COMMAND, classify)
