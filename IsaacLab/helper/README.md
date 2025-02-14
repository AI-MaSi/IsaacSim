# Helper
So much to learn, so I scraped some of the example files provided with IsaacLab, and fed them to Claudes (the AI, not sponsored haha) project. Seems to work pretty good.

Due to file size constrains, I needed to skip large chunk of examples. I guess it would be smart to create many Claude-projects that specify to each part of the development. So basically different blacklists for each part of the project

More to come.

## Usage

Thow scraper and blacklist to the main IsaacLab folder, and run

```plaintext
python scraper_to_txt.py --use-list blacklist_direct_env.txt
```

Current blacklist fills around 80% knowledge capacity.


## Claudes project prompt (could be better)

Analyze the uploaded documents and extract all available information accurately and comprehensively. 
Do not generate or infer any new information beyond what is explicitly stated in the documents. 
Maintain the original structure and wording as much as possible. 

When presenting code or structured data, mimic the programming style, syntax, and formatting conventions used in the provided examples.
