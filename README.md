# Colour Palette Analyzer & Blog Post Writer
Powered by [Site Palette](https://palette.site)

This script generates blog posts about color palettes with accompanying HTML, metadata, and OG images.

![image](https://github.com/user-attachments/assets/b65f197b-35cb-4aad-b91c-920c10e127ed)
![CleanShot 2024-10-03 at 23 58 33@2x](https://github.com/user-attachments/assets/d0c85ddf-6e12-4553-8562-fa84dbfc3c48)

## Requirements

- Python 3.7+
- Required libraries: 
  - json, collections, re, os, datetime, PIL, colorama, ell, tqdm, logging, io, replicate, requests, subprocess, importlib, ast, sys, time, asyncio, aiohttp, markdown, bleach, random, wand

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Ensure you have a valid Replicate API key set in your environment variables.

3. Place your palette data in a file named `input_output.json` in the same directory.

## Usage

Run the script:

```
python palette-post-writer.py
```

The script will:
1. Generate topic options
2. Allow you to select a topic or input a custom one
3. Generate and test a palette selection method
4. Create a blog post, HTML page, metadata, and OG image
5. Save all generated files

## Output

- HTML file with the blog post content
- JSON file with metadata
- WebP image file for OG image
- Logs in `palette_post_writer.log`

## Note

This script uses AI models for content generation. Ensure you have necessary permissions and comply with usage policies of the AI services used.
