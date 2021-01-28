## Fetch images

Install instagram-scraper

```
python3 -m pip install instagram-scraper
```

Fetch images

```
instagram-scraper --tag handsup -t image
```

## Setup filtering

Install detectron2

```
python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

## Filter

```
python3 filterByPose.py --imagesDir <images directory> --filteredDir <filtered images directory>
```