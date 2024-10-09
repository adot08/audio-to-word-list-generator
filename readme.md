# English Podcast Vocabulary Helper

🎧 Simplify your English podcast listening experience with automatic vocabulary lists

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

English Podcast Vocabulary Helper is an innovative tool designed to enhance your English learning experience by automatically generating vocabulary lists from podcasts and other audio/text sources. Say goodbye to the frustration of encountering unfamiliar words and the tedium of manual lookup!

![Web Page Screenshot](web_page.png)

## 🌟 Key Features

- **Multi-format Support**: Upload word lists in various formats (txt, md, rtf, mp3, wav, ogg, flac, json)
- **Automatic Vocabulary Generation**: Quickly create lists of challenging words from your content
- **Comprehensive Word Information**: View definitions, example sentences, and generate new contexts
- **Export Functionality**: Save your progress by exporting word data in JSON format
- **User-friendly Interface**: Easy-to-use web application for seamless interaction

## 🚀 Quick Start

1. **Clone the repository**
   ```
   git clone https://github.com/adot08/audio-to-word-list-generator.git
   cd audio-to-word-list-generato
   ```

2. **Set up the environment**
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. **Launch the application**
   ```
   python app.py
   ```

4. Open your browser and visit `http://localhost:5000`

5. Upload your audio/text file and start exploring your personalized vocabulary list!

## 📘 Detailed Usage

1. **File Upload**: Click "Choose File" to select your audio or text file.
2. **Processing**: The system will automatically process your file and generate a list of challenging words.
3. **Explore**: Browse through the generated vocabulary list, view definitions, and example sentences.
4. **Interact**: Generate new example sentences or export your word data for future reference.

## ⚙️ Configuration

All customizable options are available in `config.yaml`. You can modify the ASR (Automatic Speech Recognition) and LLM (Language Model) services according to your preferences. This project utilizes SiliconFlow services for comprehensive functionality and easy model switching.

## 📊 Performance

- Processing time may vary for larger audio files due to segmentation and ASR operations.
- AI-generated explanations might take additional time, depending on the number of challenging words identified.
- For quicker access in future sessions, processed files can be exported and later re-imported.

## 🤝 Contributing

We welcome contributions to enhance the English Podcast Vocabulary Helper! Please check out our [Contributing Guidelines](CONTRIBUTING.md) for more information on how to get started.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- Thanks to all contributors who have helped shape this project.
- Special thanks to SiliconFlow for providing comprehensive AI services.

## 📞 Contact

For any queries or suggestions, please open an issue in this repository or contact the maintainer at [your-email@example.com].

---

Keywords: English learning, podcast vocabulary, automatic word list, difficult word viewer, language learning tool, ASR, NLP