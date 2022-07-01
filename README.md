# Crag Finder
A deep learning model for classifying map tiles as containing or not containing climbing. A write up of the results is here: [http://michaelskaug.com/crag_finder/](http://michaelskaug.com/crag_finder/)

**NOTE**

There was a long interval of time between when I worked on this and when I put it on github, so there are some missing pieces that would need to be filled in if you actually wanted to reproduce the training and results. For example, there is no requirements.txt or a script for compiling the training data (although if you look at `data/training.csv` you can probably figure out how to do it.)

## Training labels
The labels for the positive class (climbing present) were derived from [MountainProject](https://www.mountainproject.com/)'s list of climbing locations. The labels for the negative class (no climbing) were based on random sampling and is described in the blog post.

## Input data
### Satellite image tiles

[Mapbox tile API](https://www.mapbox.com/api-documentation/#retrieve-tiles)

    https://api.mapbox.com/v4/mapbox.satellite/3/2/3.jpg90?access_token=your-access-token

returns a 256x256 pixel map tile

### Terrain tiles

[Mapzen tiles via AWS](https://mapzen.com/documentation/terrain-tiles/use-service/)

    https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png

returns a 256x256 pixel tile with elevation encoded in rgb channels

### Street tiles

Maybe use [Mapbox high-contrast](http://api.mapbox.com/v4/mapbox.high-contrast.html?access_token=pk.eyJ1IjoibWFwYm94IiwiYSI6ImNpejY4NDg1bDA1cjYzM280NHJ5NzlvNDMifQ.d6e-nNyBDtmQCVwVNivz7A#3/0.00/0.00)?

[Mapbox tile API](https://www.mapbox.com/api-documentation/#retrieve-tiles)

    https://api.mapbox.com/v4/mapbox.high-contrast/3/2/3.jpg90?access_token=your-access-token"

returns a 256x256 pixel map tile

### Deep Learning AMI

[DLAMI](https://docs.aws.amazon.com/dlami/latest/devguide/what-is-dlami.html)
