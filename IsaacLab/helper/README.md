# Helper
So much to learn, so I scraped some of the example files provided with IsaacLab, and fed them to Claudes project. Seems to work pretty good. Also scrapes the configclasses, so Claude can understand what to use and so on.

Due to file size constrains, I needed to skip large chunk of examples. I guess it would be smart to create many Claude-projects that specify to each part of the development. so basically different black lists for each part of the project

More to come, very raw system still.

## Usage

```plaintext
scraper.py --select-list base_blacklist.txt --max-size 1000
```

selects base_blacklist.txt and sets maximum output file size to 1000KB