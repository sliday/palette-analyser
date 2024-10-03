import json
from collections import Counter
import re
import os
from datetime import datetime
from PIL import Image, ImageDraw
import colorama
from colorama import Fore, Style
import ell
from tqdm import tqdm
import logging
import io
import replicate
import requests
import subprocess
from datetime import datetime, timedelta
import importlib
import ast
import sys
import time
import asyncio
import aiohttp
import logging
import time
import traceback
import markdown
import bleach
import random
from wand.image import Image as WandImage
from wand.drawing import Drawing
from wand.color import Color

# Initialize colorama for colored terminal output
colorama.init(autoreset=True)

# Initialize ell
ell.init(store='./logdir', autocommit=True, verbose=True)

# Set up logging
logging.basicConfig(filename='palette_post_writer.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load JSON data
with open('colours.json', 'r') as f:
    palettes = json.load(f)

@ell.tool()


def process_palettes(prompt: str):
    logging.info(f"Processing palettes for prompt: {prompt}")
    
    # Simple parsing to extract month and year from prompt
    import re
    match = re.search(r'from (\w+) (\d{4})', prompt)
    if match:
        target_month = match.group(1)
        target_year = int(match.group(2))
    else:
        target_month = None
        target_year = None

    # Filter palettes based on time
    if target_month and target_year:
        filtered_palettes = [
            p for p in palettes
            if p['time_based']['month'] == target_month and p['time_based']['year'] == target_year
        ]
    else:
        filtered_palettes = palettes  # If no time specified, return all

    # Use gpt-4o-mini for reasoning per palette
    reasoned_palettes = []
    for palette in filtered_palettes:
        try:
            reasoned_palette = analyze_palette(json.dumps(palette), prompt, topic) # type: ignore
            # Attempt to parse the result as JSON, if it fails, use the original palette
            try:
                parsed_palette = json.loads(reasoned_palette)
                reasoned_palettes.append(parsed_palette)
            except json.JSONDecodeError:
                logging.warning(f"Failed to parse analysis for palette {palette['palette_id']}. Using original palette.")
                reasoned_palettes.append(palette)
        except Exception as e:
            logging.error(f"Error analyzing palette {palette['palette_id']}: {str(e)}")
            reasoned_palettes.append(palette)

    logging.info(f"Processed {len(reasoned_palettes)} palettes")
    return json.dumps(reasoned_palettes)

json_schema = """
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "palette_id": { "type": "integer" },
      "title": { "type": "string" },
      "emoji": { "type": "string" },
      "colors": {
        "type": "object",
        "properties": {
          "hex_values": {
            "type": "array",
            "items": { "type": "string" }
          },
          "color_names": {
            "type": "array",
            "items": { "type": "string" }
          }
        }
      },
      "color_properties": {
        "type": "object",
        "properties": {
          "dominant_colors": {
            "type": "array",
            "items": { "type": "string" }
          },
          "color_schemes": {
            "type": "array",
            "items": { "type": "string" }
          },
          "warmth_coolness": { "type": "string" },
          "brightness_level": { "type": "string" },
          "saturation_level": { "type": "string" }
        }
      },
      "themes": {
        "type": "object",
        "properties": {
          "mood": { "type": "string" },
          "style": { "type": "string" },
          "season": { "type": "string" },
          "occasion": { "type": "string" }
        }
      },
      "application_contexts": {
        "type": "object",
        "properties": {
          "suitable_rooms": {
            "type": "array",
            "items": { "type": "string" }
          },
          "interior_design_styles": {
            "type": "array",
            "items": { "type": "string" }
          },
          "industries": {
            "type": "array",
            "items": { "type": "string" }
          },
          "projects": {
            "type": "array",
            "items": { "type": "string" }
          }
        }
      },
      "time_based": {
        "type": "object",
        "properties": {
          "creation_timestamp": { "type": "integer" },
          "month": { "type": "string" },
          "year": { "type": "integer" }
        }
      }
    },
    "required": ["palette_id", "title", "emoji", "colors", "color_properties", "themes", "application_contexts", "time_based"]
  }
}
"""

@ell.simple(model="gpt-4o-mini", max_tokens=4000)
def analyze_palette(palette_json: str, prompt: str, topic: str):
    """You are an expert color palette analyzer. Analyze the given palette and provide insights rooter in {prompt} ({topic}).
    Return your analysis as a JSON string with the following structure:
    {
        "original_palette": <original palette JSON>,
        "analysis": <your analysis as a string>
    }
    No intro, no commentary, no explanations. Only the required content in required format.
    """
    return f"Analyze this palette and provide insights: {palette_json}"

@ell.simple(model="claude-3-5-sonnet-20240620", max_tokens=4000)
def generate_blog_article(prompt: str, palettes_json: str, topic: str):
    """You are an expert color palette analyst and markdown blog writer."""
    with open(palettes_json, 'r') as f:
        palettes = json.load(f)
    
    return f"""Write a detailed blog article about the following color palettes, must use the exact palette names and emojis, colors, and properties provided in the JSON data: {json.dumps(palettes)}. 
The user asked: '{prompt}'. The specific topic is: '{topic}'.
Provide in-depth analysis of the palettes, discuss their properties, and explain how they can be utilized in the context of {topic}.
Ensure the article is engaging and informative. Mimic NYT style. Keep it short and concise: no more than 1000 words.
Must avoid using AI words: Delve, Harnessing, At the heart of, In essence, Facilitating, Intrinsic, Integral, Core, Facet, Nuance, Culmination, Manifestation, Inherent, Confluence, Underlying, Intricacies, Epitomize, Embodiment, Iteration, Synthesize, Amplify, Impetus, Catalyst, Synergy, Cohesive, Paradigm, Dynamics, Implications, Prerequisite, Fusion, Holistic, Quintessential, Cohesion, Symbiosis, Integration, Encompass, Unveil, Unravel, Emanate, Illuminate, Reverberate, Augment, Infuse, Extrapolate, Embody, Unify, Inflection, Instigate, Embark, Envisage, Elucidate, Substantiate, Resonate, Catalyze, Resilience, Evoke, Pinnacle, Evolve, Digital Bazaar, Tapestry
Leverage, Centerpiece, Subtlety, Immanent, Exemplify, Blend, Comprehensive, Archetypal, Unity, Harmony
Conceptualize, Reinforce, Mosaic, Catering.

Use the exact palette names and emojis, colors, and properties provided in the JSON data. You can only speak Markdown.
"""

@ell.simple(model="claude-3-5-sonnet-20240620", max_tokens=2000)
def design_palette_selection_method(prompt: str, topic: str, previous_code: str = ""):
    return f"""
    You are an expert in python, data processing and statistical analysis. Design a method to select the top 10 most relevant color palettes from a large JSON file based on a given prompt and topic. The method should be simple, concise, and implementable in Python.

    Design a innovative method to select the top 10 most relevant color palettes from a large JSON file containing 1000+ palettes, based on the following:
    
    Note sklearn is deprecated, use scikit-learn if required.

JSON file structure is pre-defined, use that to extract the data you need. Sample object (in yaml for readability):
```
{json_schema}
```

    User prompt: {prompt}
    Methodology (extra details): {topic}
    
    The python script code method MUST:
    1. Be ultra-simple, straightforward and concise
    2. Use condensed statistical and text analysis techniques
    3. Be implementable in ultra-short vanilla Python code
    4. MUST be relevant to both the prompt and the methodology.
    5. Output a list of 10 palette palette IDs in python array of integers.
    6. Be able to run as a separate Python script on a local machine, production-ready
    7. MUST include a function named `select_top_palettes` that takes one parameter:
       - json_file: str (path to the JSON file)
       This function should return the list of selected palette IDs, maximum 10.
    
    Previous code (if any):
    {previous_code}

    If there was an error in the previous attempt, please analyze it and make necessary improvements.

    No intro, no commentary, no explanations. Only the required production-ready Python code.
    Strictly No \`\`\`python\`\`\`!
    """


@ell.simple(model="claude-3-5-sonnet-20240620", max_tokens=8192)
def generate_html(article: str, selected_palettes_json: str):
    """You are an expert HTML and minimalistic Tailwind CSS developer.
    No intro, no commentary, no explanations. Only the required content in required format.
    """
    logging.info("Generating HTML content")
    
    # Convert markdown to HTML
    html_content = markdown_to_html(article)
    
    return f"""Improve and enhance the following HTML content using Tailwind CSS and Tailwind Typography:

{html_content}

MUST Include large and pretty colored flex blocks to display the palettes provided: 
```
{selected_palettes_json}
```

Add vanilla js to allow to copy hex values to clipboard when I click on the blocks.

The main content within a 'prose' container. Palette blocks inserted in-between: colored blocks with titles, emojis and as many as possible hex_values as background color. The article is about colours afterall.


Ensure the page is responsive, visually appealing, and follows best practices.
No intro, no commentary, no explanations. Only the required content in required format.
No \`\`\`html\`\`\`!
"""

def markdown_to_html(markdown_text: str) -> str:
    """Convert markdown to HTML and sanitize the output."""
    # Convert markdown to HTML
    html = markdown.markdown(markdown_text)
    
    # Sanitize the HTML to remove any potentially harmful content
    allowed_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'a', 'ul', 'ol', 'li', 'strong', 'em', 'blockquote', 'code', 'pre']
    allowed_attributes = {'a': ['href', 'title']}
    
    clean_html = bleach.clean(html, tags=allowed_tags, attributes=allowed_attributes)
    
    return clean_html

@ell.simple(model="gpt-4o", max_tokens=500)
def generate_metadata(title: str, content: str):
    """You are an SEO expert specializing in color palettes and design.
    No intro, no commentary, no explanations. Only the required content in required format.
    """
    return f"""Generate metadata for SEO purposes for an article about home decor. Provide a title, description, and keywords based on the content below:
{title}
{content[:500]}...
Ensure the metadata is concise and effective. Return the result as a JSON object with keys: title, description, and keywords.
No intro, no commentary, no explanations. Only the required content in required format.
No \`\`\`json\`\`\`!
"""

@ell.simple(model="gpt-4o-mini")
def generate_og_image_prompt(title: str, content: str, palettes_json: str):
    """You are an expert at creating concise, visually descriptive prompts for image generation."""
    return f"""Create a prompt for an OG image based on this blog post:
Title: {title}
Content: {content[:2000]}...
Palettes: {palettes_json}

The prompt should be simple, visual, follow Stable Diffusion latest guidelines and under 500 characters. Focus on key elements that represent the blog's topic.

No intro, no commentary, no explanations. 
No \`\`\`json\`\`\`! Plain text, simple English prompt. Only the required content in required format.
"""

async def create_og_image(palettes, title, content):
    try:
        # Generate prompt using all selected palettes
        prompt = generate_og_image_prompt(title, content[:2000], json.dumps(palettes))
        logging.info(f"Generated prompt: {prompt}")

        # Run Replicate prediction synchronously
        logging.info("Starting Replicate prediction")
        output = replicate.run(
            "black-forest-labs/flux-1.1-pro",
            input={
                "prompt": prompt,
                "aspect_ratio": "16:9",
                "output_format": "webp",
                "output_quality": 80,
                "safety_tolerance": 2,
                "prompt_upsampling": True
            }
        )

        logging.info(f"Replicate output: {output}")

        # Use the output directly as the image URL
        image_url = output[0] if isinstance(output, list) else output
        logging.info(f"Image URL: {image_url}")

        # Download the generated image asynchronously
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as response:
                response.raise_for_status()
                img_data = await response.read()

        logging.info("Image successfully downloaded")

    except Exception as e:
        logging.error(f"Error generating or downloading image: {str(e)}")
        # Create a default image if generation or download fails
        img_data = create_default_image()
        logging.info("Created default image due to failure")

    # Add caption to the image
    img_with_caption = add_caption_to_image(img_data, title)

    logging.info("Caption added to image")
    return img_with_caption

def create_default_image():
    with WandImage(width=1200, height=630, background=Color('white')) as img:
        return img.make_blob('webp')

def add_caption_to_image(img_data, title):
    # Remove "# " from the beginning of the title if present
    title = title.lstrip('# ')

    # Add line break in the middle of the text
    words = title.split()
    mid = len(words) // 2
    title = ' '.join(words[:mid]) + '\n' + ' '.join(words[mid:])

    with WandImage(blob=img_data) as img:
        width, height = img.width, img.height
        
        with Drawing() as draw:
            # Set up the text properties
            font_family = 'Arial'
            draw.font_family = font_family
            draw.font_style = 'normal'
            draw.font_weight = 700
            draw.fill_color = Color('white')
            draw.text_kerning = -5  # -5% letter-spacing
            
            # Find the largest font size that fits (1.6x larger than before)
            font_size = int(min(width, height) // 3.125)  # Start with a larger font size (5 / 1.6 ≈ 3.125)
            draw.font_size = font_size
            metrics = draw.get_font_metrics(img, title, multiline=True)
            
            while metrics.text_width > width * 0.9 or metrics.text_height > height * 0.9:
                font_size -= 1
                draw.font_size = font_size
                metrics = draw.get_font_metrics(img, title, multiline=True)
            
            # Calculate position to center the text
            x = max(0, (width - int(metrics.text_width)) // 2)
            y = max(0, (height - int(metrics.text_height)) // 2 + int(metrics.ascender))
            
            # Draw shadow
            shadow_offset = max(10, font_size // 20)  # Adjust shadow offset based on font size
            with Drawing() as shadow:
                shadow.font_family = font_family
                shadow.font_size = font_size
                shadow.font_style = 'normal'
                shadow.font_weight = 700
                shadow.text_kerning = -5  # -5% letter-spacing
                shadow.fill_color = Color('rgba(0, 0, 0, 0.015)')  # Semi-transparent black
                for dx in range(-shadow_offset, shadow_offset + 1, 2):
                    for dy in range(-shadow_offset, shadow_offset + 1, 2):
                        shadow.text(x + dx, y + dy, title)
                shadow(img)
            
            # Draw main text
            draw.text(x, y, title)
            
            # Apply the drawing to the image
            draw(img)
        
        return img.make_blob('webp')

def save_metadata(metadata, filename='metadata.json'):
    with open(filename, 'w') as f:
        json.dump(metadata, f, indent=2)
    logging.info(f"Metadata saved to {filename}")

def save_files(html_content, metadata, og_image, title):
    # Sanitize the title for use as a filename
    safe_title = "".join([c for c in title if c.isalpha() or c.isdigit() or c==' ']).rstrip()
    safe_title = safe_title.replace(' ', '_').lower()
    
    # Save HTML file
    html_filename = f'{safe_title}.html'
    with open(html_filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"{Fore.GREEN}✅ HTML file saved as {html_filename}{Style.RESET_ALL}")

    # Save metadata as JSON
    metadata_filename = f'{safe_title}_metadata.json'
    with open(metadata_filename, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"{Fore.GREEN}✅ Metadata saved as {metadata_filename}{Style.RESET_ALL}")

    # Save OG image as WebP
    og_image_filename = f'{safe_title}_og-image.webp'
    with open(og_image_filename, 'wb') as f:
        f.write(og_image)
    print(f"{Fore.GREEN}✅ OG image saved as {og_image_filename}{Style.RESET_ALL}")

PALETTE_ID = "palette_id"
TITLE = "title"
EMOJI = "emoji"
COLORS = "colors"
HEX_VALUES = "hex_values"
COLOR_NAMES = "color_names"

@ell.simple(model="gpt-4o", max_tokens=1000, temperature=1)
def generate_topic_options():
    """You are a color enthusiast tasked with generating random ideas for color palettes posts."""
    prompt = f"""
Random seed {random.randint(0, 100000)}. Generate 3 unique pairs of titles and methodologies for articles about color palette trends.

IMPORTANT: Get creative. Play with the palette data. Get unorthodox. Surprise me every time! Explore colours in these categories:
- top 10 most popular palettes
- top 10 most popular colors
- top 10 most popular color combinations
- top 10 most popular color schemes
- top 10 most popular color trends
- top 10 most popular color psychology
- top 10 most popular color symbolism
- top 10 most popular color combinations
- top 10 most popular color trends
- decoration
- design
- art
- fashion
- architecture
- nature
- technology
- food
- emotions
- psychology
- science
- history
- culture
- travel
- health
- education
- business
- marketing
- sports
- entertainment
- literature
- music
- film
- photography
- interior design
- random

You can calculate the trend by using the palette data in any way you like: average, median, mode, range, standard deviation, max, min, top 3 most common, etc. You can also use the color properties, themes, application contexts, and time-based properties.

Each pair should consist of:
- Interesting title
- A brief, but actionable description of the methodology to calculate or determine the trend

Use these vars:
PALETTE_ID = "palette_id"
TITLE = "title"
EMOJI = "emoji"
COLORS = "colors"
HEX_VALUES = "hex_values"
COLOR_NAMES = "color_names"

When accessing JSON data, use .get() method or try-except blocks to handle potentially missing keys:
palette_id = palette.get(PALETTE_ID, None)
hex_values = palette.get(COLORS, {{}}).get(HEX_VALUES, [])

Methodology should take into consideration the following vars available in the JSON file (showed as yaml for readability):
```
{json_schema}
```

Output the results in JSON format, with each pair as an object containing "title" and "methodology" keys.

```
{{
  "trend_pairs": [
    {{
      "title": "string",
      "methodology": "string"
    }},
    ...
  ]
}}
```

No intro, no commentary, no explanations. Only the required content in JSON format.
No \`\`\`json\`\`\`!
"""
    return prompt

def detect_imports(file_path):
    with open(file_path, 'r') as file:
        tree = ast.parse(file.read())
    
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module)
    
    return imports

def install_dependencies(dependencies):
    for dep in dependencies:
        try:
            importlib.import_module(dep)
        except ImportError:
            print(f"Installing {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])

def generate_and_test_selector(prompt, topic, max_attempts=3):
    previous_code = ""
    for attempt in range(max_attempts):
        try:
            print(f"{Fore.YELLOW}🔍 Designing palette selection method (Attempt {attempt + 1}/{max_attempts})...{Style.RESET_ALL}")
            
            # Generate the method description/code, including previous code and error if available
            method_description = design_palette_selection_method(prompt, topic, previous_code)
            print(method_description)
            
            # Save the generated code to a temporary file
            with open('generated_palette_selector.py', 'w') as f:
                f.write(method_description)
            
            # Store the current code for the next iteration if needed
            previous_code = method_description
            
            # Detect and install dependencies
            print(f"{Fore.YELLOW}📦 Detecting and installing dependencies...{Style.RESET_ALL}")
            dependencies = detect_imports('generated_palette_selector.py')
            install_dependencies(dependencies)
            
            # Import the generated method
            if 'generated_palette_selector' in sys.modules:
                del sys.modules['generated_palette_selector']
            import generated_palette_selector
            importlib.reload(generated_palette_selector)
            
            # Test the generated method
            selected_palette_ids = generated_palette_selector.select_top_palettes('colours.json')
            
            # Check if the result is valid
            if isinstance(selected_palette_ids, list) and len(selected_palette_ids) > 0 and all(isinstance(pid, int) for pid in selected_palette_ids):
                print(f"{Fore.GREEN}✅ Successfully generated and tested palette selector.{Style.RESET_ALL}")
                return selected_palette_ids
            else:
                raise ValueError("Invalid output from select_top_palettes")
        
        except Exception as e:
            error_message = f"Error in attempt {attempt + 1}: {str(e)}\n{traceback.format_exc()}"
            print(f"{Fore.RED}❌ {error_message}{Style.RESET_ALL}")
            if attempt == max_attempts - 1:
                print(f"{Fore.RED}❌ Failed to generate a working palette selector after {max_attempts} attempts.{Style.RESET_ALL}")
                return None
            else:
                # Include the error message in the previous_code for the next iteration
                previous_code += f"\n\n# Error from previous attempt:\n'''{error_message}'''\n"

    return None

async def main():
    
    print(f"{Fore.CYAN}🎨 Welcome to the Palette Post Writer!{Style.RESET_ALL}")
    
    # Generate topic options
    print(f"{Fore.YELLOW}🔖 Generating topic options...{Style.RESET_ALL}")
    topic_options_json = generate_topic_options()
    
    # Parse the generated JSON
    try:
        topic_options = json.loads(topic_options_json)
    except json.JSONDecodeError:
        print(f"{Fore.RED}❌ Failed to parse topic options. Please provide custom inputs.{Style.RESET_ALL}")
        topic_options = {'trend_pairs': []}
    
    # Check if any topics were generated
    if topic_options.get('trend_pairs'):
        print(f"{Fore.GREEN}Here are some topic options for you to choose from:{Style.RESET_ALL}")
        for idx, pair in enumerate(topic_options['trend_pairs'], 1):
            print(f"{idx}. {pair['title']}")
        selection = input(f"{Fore.GREEN}👉 Enter the number of the topic you want to select (press Enter to skip): {Style.RESET_ALL}")
        if selection.strip().isdigit() and 1 <= int(selection) <= len(topic_options['trend_pairs']):
            selected_pair = topic_options['trend_pairs'][int(selection) - 1]
            prompt = selected_pair['title']
            topic = selected_pair['methodology']
        else:
            # Custom input if selection is invalid or skipped
            prompt = input(f"{Fore.GREEN}👉 Enter your custom post title: {Style.RESET_ALL}")
            topic = input(f"{Fore.GREEN}📝 Explain (vaguely) how the palettes should be selected: {Style.RESET_ALL}")
    else:
        # Fallback to custom input if no topics generated
        prompt = input(f"{Fore.GREEN}👉 Enter your post title: {Style.RESET_ALL}")
        topic = input(f"{Fore.GREEN}📝 Explain (vaguely) how the palettes should be selected: {Style.RESET_ALL}")
    
    try:
        selected_palette_ids = generate_and_test_selector(prompt, topic)
        
        if selected_palette_ids is None:
            print(f"{Fore.RED}❌ Unable to generate a working palette selector. Exiting.{Style.RESET_ALL}")
            return

        print(f"{Fore.YELLOW}🔍 Selected palette IDs: {selected_palette_ids}{Style.RESET_ALL}")
        
        # Load the palettes JSON
        with open('colours.json', 'r') as f:
            all_palettes = json.load(f)
        
        # Filter the palettes based on selected IDs
        filtered_palettes = [p for p in all_palettes if p['palette_id'] in selected_palette_ids]
        
        # Save the selected palettes to a file
        with open('selected_palettes.json', 'w') as f:
            json.dump(filtered_palettes, f, indent=2)
        
        if not filtered_palettes:
            print(f"{Fore.RED}❌ No palettes found matching the criteria.{Style.RESET_ALL}")
            return

        
        print(f"{Fore.YELLOW}🖊️ Generating blog article...{Style.RESET_ALL}")
        article = generate_blog_article(prompt, 'selected_palettes.json', topic)
        
        print(f"{Fore.YELLOW}💻 Creating HTML page...{Style.RESET_ALL}")
        html_content = generate_html(article, json.dumps(filtered_palettes, indent=2))
        
        print(f"{Fore.YELLOW}📄 Generating metadata...{Style.RESET_ALL}")
        metadata = generate_metadata(article.split('\n')[0], article)
        
        print(f"{Fore.YELLOW}🎨 Creating OG image...{Style.RESET_ALL}")
        og_image = await create_og_image(filtered_palettes, article.split('\n')[0], article)
        
        print(f"{Fore.YELLOW}💾 Saving files...{Style.RESET_ALL}")
        save_files(html_content, metadata, og_image, prompt)  # Use 'prompt' as the title
        
        print(f"{Fore.GREEN}✅ All tasks completed successfully!{Style.RESET_ALL}")
        
    except Exception as e:
        print(f"{Fore.RED}❌ An error occurred: {str(e)}{Style.RESET_ALL}")
        print(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main())