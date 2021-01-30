{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import json\n",
    "from pprint import pprint\n",
    "import os\n",
    "import PIL.ImageDraw as ImageDraw\n",
    "import PIL.Image as Image\n",
    "\n",
    "\n",
    "kartinki=[]\n",
    "foo='/...Path to folder Supervise Person Dataset../img/'\n",
    "for root, dirs, files in os.walk(\"/Users/grigorijasaev/Downloads/ODS/ds0/ann/\"):  \n",
    "    for filename in files:\n",
    "        print(foo+filename[:-5])\n",
    "        kartinki.append(foo+filename[:-5])\n",
    "\n",
    "for root, dirs, files in os.walk(\"/...Path to folder Supervise Person Dataset../ann/\"):  \n",
    "    for ch,filename in enumerate(files):\n",
    "        with open(root+filename) as json_data:\n",
    "            d = json.loads(json_data.read())\n",
    "            json_data.close()\n",
    "        if 'bitmap' not in str(d):\n",
    "            print(filename)\n",
    "            a=d['objects'][0]['points']['exterior'] \n",
    "            image = Image.new(\"RGB\", (d['size']['width'],d['size']['height']))\n",
    "            draw = ImageDraw.Draw(image)\n",
    "            points = sum(a,[])\n",
    "            draw.polygon((points), fill='WHITE')\n",
    "            image.show()\n",
    "            im1 = image.save(filename+'.png') \n",
    "            imagew = Image.open(kartinki[ch])\n",
    "            print(kartinki[ch])\n",
    "            imagew.show()\n",
    "            im2 = imagew.save(filename+'mask'+'.png') \n",
    "        else:\n",
    "            print('Error')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
