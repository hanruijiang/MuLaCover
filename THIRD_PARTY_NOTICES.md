# Third-Party Notices

MuLaCover uses the following third-party transcription components:

- [YourMT3](https://github.com/mimbres/YourMT3) — melody and percussion transcription. See [source notices and licenses](src/mulacover/_symbolic_transcription/yourmt3/THIRD_PARTY.md).
- [Music X Lab chord recognition](https://github.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition) — chord transcription. See [MIT license](src/mulacover/_symbolic_transcription/chord/LICENSE) and [notice](src/mulacover/_symbolic_transcription/chord/NOTICE.md).

Third-party components remain subject to their respective licenses. The YourMT3 upstream repository uses GPL-3.0 while the adapted source files carry Apache-2.0 headers; the applicable source revision and authorization require verification before public distribution.

Python dependencies are declared in [pyproject.toml](pyproject.toml).
