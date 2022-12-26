# SPICED Week04: Scrape lyrics of two artists from www.lyrics.com and train a model to distinguish between them using the 'Bag of Words' method

## Project Summary

In this project, I build a text classification model on song lyrics of two different artists. At the end, the task is to determine which of the two artists has written a particular song. To train such a model, the script will scrape the lyrics from the web. The following steps are necessary to achive the project goal:

- Download a HTML page with links to songs
- Extract hyperlinks of song pages
- Download and extract the song lyrics
- Vectorize the text using the Bag Of Words method
- train a classification model that predicts the artist from a piece of text
- refactor the code into functions

## Documentation

Frontend visualization of the script 'lyrics_exe.command' for the artists 'Frightened Rabbit' and 'Gaslight Anthem':

![Bildschirmfoto 2022-12-26 um 16 15 32](https://user-images.githubusercontent.com/61935581/209565168-60548b75-13f4-4d16-a233-45f292db681f.png)

The results with the test data were not satisfactory, which could be due to the fact that both bands use a similar language and the corpus of song lyrics studied was comparatively small with 20 songs each.

## Usage

Clone the repository and double click on 'lyrics_exe.command'. The script will then prompt you for the page of artist 1 and 2 on lyrics.com.
In the example shown above, these were the following pages:

- 'Frightened-Rabbit/807408'
- 'The-Gaslight-Anthem/909493'

Afterwards, the script will download the lyrics and ask the user for the size of the corpus. Be aware that the corpus can not be bigger than the amount of downloaded lyrics.

Finally, for models will be trained using the 'Bag of Words' method and their results displayed in the terminal.

## Additional Material

In the Jupiter notebooks, the step-by-step development of the Python script can be retraced. In addition, they contain a few more features such as the creation of a wordcloud (see 'WP4_Pipeline.ipynb').

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
