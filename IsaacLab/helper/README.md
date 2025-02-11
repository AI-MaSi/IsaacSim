# Helper
So much to learn, so I scraped some of the example files provided with IsaacLab, and fed them to Claudes (the AI, not sponsored haha) project. Seems to work pretty good. Also scraped the configclasses, so Claude can understand what to use and so on.

Due to file size constrains, I needed to skip large chunk of examples. I guess it would be smart to create many Claude-projects that specify to each part of the development. So basically different blacklists for each part of the project

More to come, very raw system still.

## Usage

```plaintext
scraper.py --select-list base_blacklist.txt --max-size 1000
```

selects base_blacklist.txt and sets maximum output file size to 1000KB.

Base blacklist focuses on directRLenvs and basic environment creation.

## Claudes project instructions
Could be better:

Analyze the uploaded documents and extract all available information accurately and comprehensively. 
Do not generate or infer any new information beyond what is explicitly stated in the documents. 
Maintain the original structure and wording as much as possible. 

When presenting code or structured data, mimic the programming style, syntax, and formatting conventions used in the provided examples.