import telebot
from telebot import types
import whisper
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor
import torch
import torchaudio
import librosa
import librosa.display
import noisereduce as nr
import matplotlib.pyplot as plt


# ссылка на бот https://t.me/Pasttime_Bot

TOKEN = '6547232592:AAGbqZoIFcH1Rp3-PJ3U2smmnl-GXqqRLp0'

# для старта игры необходимо запустить этот скрипт и необходимо вставить правильную ссылку на файл game.json в вашей дирректории


bot = telebot.TeleBot(TOKEN)


# команда, которая инициирует начало диалога с ботом
@bot.message_handler(commands=['start'])
def start(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    item1 = types.InlineKeyboardButton('Меню')
    # item2 = types.InlineKeyboardButton('Вопросы и пожелания')

    markup.add(item1)

    bot.send_message(message.chat.id, 'Привет, {0.first_name}!'.format(message.from_user), reply_markup=markup)

def create_waveplot(data, sr, e):
    plt.figure(figsize=(10, 3))
    plt.title('Waveplot for audio with {} emotion'.format(e), size=15)
    librosa.display.waveshow(data, sr=sr, color="blue")
    plt.savefig('wave.png')

def create_spectrogram(data, sr, e):
    X = librosa.stft(data)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(10, 3))
    plt.title('Spectrogram for audio with {} emotion'.format(e), size=15)
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz', cmap='magma')
    plt.savefig('specto.png')

@bot.message_handler(commands=['send_image'])
def send_image(message):
    # Путь к картинке на сервере
    image_path = 'wave.png'

    # Открываем файл и отправляем его пользователю с помощью метода send_photo
    with open(image_path, 'rb') as photo:
        bot.send_photo(message.chat.id, photo)

@bot.message_handler(commands=['send_image1'])
def send_image(message):
    # Путь к картинке на сервере
    image_path = 'specto.png'

    # Открываем файл и отправляем его пользователю с помощью метода send_photo
    with open(image_path, 'rb') as photo:
        bot.send_photo(message.chat.id, photo)

@bot.message_handler(content_types=['text'])
def bot_message(message):
    global pred
    # if message.chat.type == 'private':
    # условие для работы с текстом меню и вернуться в меню
    if message.text == 'Меню':
        # создается и настраиваться клавиатуру для бота, которая автоматически растягивается по ширине экрана для удобства пользователя.
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        # реализация кнопок
        button0 = types.InlineKeyboardButton("Запись аудио")
        button1 = types.InlineKeyboardButton("Транскрибация")
        button2 = types.InlineKeyboardButton("Прогноз")
        button3 = types.InlineKeyboardButton("Волновая диаграмма")
        button4 = types.InlineKeyboardButton("Спектограмма")
        button5 = types.InlineKeyboardButton("Меню")
        markup.add(button0, button1, button2, button3, button4, button5)
        # выведение кнопок и сообщения
        bot.send_message(message.chat.id, 'Выбери действие'.format(message.from_user), reply_markup=markup)
    elif message.text == 'Запись аудио':
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        button0 = types.InlineKeyboardButton("Запись аудио")
        button1 = types.InlineKeyboardButton("Транскрибация")
        button2 = types.InlineKeyboardButton("Прогноз")
        button3 = types.InlineKeyboardButton("Волновая диаграмма")
        button4 = types.InlineKeyboardButton("Спектограмма")
        button5 = types.InlineKeyboardButton("Меню")
        markup.add(button0, button1, button2, button3, button4, button5)
        bot.send_message(message.chat.id, 'Запиши аудио'.format(message.from_user), reply_markup=markup)
    elif message.text == 'Транскрибация':
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        model = whisper.load_model('base')
        result = model.transcribe('audio.wav')
        print(result["text"])
        button0 = types.InlineKeyboardButton("Запись аудио")
        button1 = types.InlineKeyboardButton("Транскрибация")
        button2 = types.InlineKeyboardButton("Прогноз")
        button3 = types.InlineKeyboardButton("Волновая диаграмма")
        button4 = types.InlineKeyboardButton("Спектограмма")
        button5 = types.InlineKeyboardButton("Меню")
        markup.add(button0, button1, button2, button3, button4, button5)
        bot.send_message(message.chat.id, result["text"].format(message.from_user), reply_markup=markup)
    elif message.text == 'Прогноз':
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)

        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-large-ls960-ft")
        model = HubertForSequenceClassification.from_pretrained(
            "xbgoose/hubert-speech-emotion-recognition-russian-dusha-finetuned")
        num2emotion = {0: 'neutral', 1: 'angry', 2: 'positive', 3: 'sad'}

        filepath = 'audio.wav'

        waveform, sample_rate = torchaudio.load(filepath, normalize=True)
        transform = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = transform(waveform)

        inputs = feature_extractor(
            waveform,
            sampling_rate=feature_extractor.sampling_rate,
            return_tensors="pt",
            padding=True,
            max_length=16000 * 10,
            truncation=True
        )

        logits = model(inputs['input_values'][0]).logits
        predictions = torch.argmax(logits, dim=-1)
        pred = num2emotion[predictions.numpy()[0]]

        button0 = types.InlineKeyboardButton("Запись аудио")
        button1 = types.InlineKeyboardButton("Транскрибация")
        button2 = types.InlineKeyboardButton("Прогноз")
        button3 = types.InlineKeyboardButton("Волновая диаграмма")
        button4 = types.InlineKeyboardButton("Спектограмма")
        button5 = types.InlineKeyboardButton("Меню")
        markup.add(button0, button1, button2, button3, button4, button5)
        bot.send_message(message.chat.id, pred.format(message.from_user), reply_markup=markup)

    elif message.text == 'Волновая диаграмма':
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        emo = pred

        data, sampling_rate = librosa.load('audio.wav')
        data = nr.reduce_noise(data, sr=sampling_rate)
        xt, index = librosa.effects.trim(data, top_db=33)
        create_waveplot(xt, sampling_rate, emo)

        button0 = types.InlineKeyboardButton("Запись аудио")
        button1 = types.InlineKeyboardButton("Транскрибация")
        button2 = types.InlineKeyboardButton("Прогноз")
        button3 = types.InlineKeyboardButton("Волновая диаграмма")
        button4 = types.InlineKeyboardButton("Спектограмма")
        button5 = types.InlineKeyboardButton("Меню")
        markup.add(button0, button1, button2, button3, button4, button5)
        bot.send_message(message.chat.id, "Нажми /send_image".format(message.from_user), reply_markup=markup)

    elif message.text == 'Спектограмма':
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)

        emo = pred

        data, sampling_rate = librosa.load('audio.wav')
        data = nr.reduce_noise(data, sr=sampling_rate)
        xt, index = librosa.effects.trim(data, top_db=33)
        create_spectrogram(xt, sampling_rate, emo)

        button0 = types.InlineKeyboardButton("Запись аудио")
        button1 = types.InlineKeyboardButton("Транскрибация")
        button2 = types.InlineKeyboardButton("Прогноз")
        button3 = types.InlineKeyboardButton("Волновая диаграмма")
        button4 = types.InlineKeyboardButton("Спектограмма")
        button5 = types.InlineKeyboardButton("Меню")
        markup.add(button0, button1, button2, button3, button4, button5)
        bot.send_message(message.chat.id, "Нажми /send_image1".format(message.from_user), reply_markup=markup)

@bot.message_handler(content_types=['voice'])
def handle_voice(message):
    # Получаем информацию о файле
    file_info = bot.get_file(message.voice.file_id)
    file_path = file_info.file_path

    # Загружаем аудиофайл
    audio = bot.download_file(file_path)

    # Сохраняем аудиофайл в формате WAV
    wav_filename = "audio.wav"
    with open(wav_filename, 'wb') as wav_file:
        wav_file.write(audio)

    # Отправляем пользователю сообщение с записанным файлом
    bot.reply_to(message, "Аудиофайл успешно записан в формате WAV.")
    with open(wav_filename, 'rb') as wav_file:
        bot.send_audio(message.chat.id, wav_file)


bot.polling(non_stop=True)