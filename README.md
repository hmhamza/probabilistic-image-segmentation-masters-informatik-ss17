# Probabilistic Image Segmentation

It was a group project done under the course **Project Hot Topics in Computer Vision** of Masters (Informatik) at TU Berlin in Summer Semester 2017.

### Abstract
We proposed an image segmentation method
based on a probabilistic model and the use of different
image features. Our region growing algorithm combines
several cues to subdivide the image into meaningful
parts. The different cues are combined and weighted
in a probabilistic model based on a fast kernel density
estimation. The probabilistic model uses the naive Bayes
theorem to combine the cues. It is therefore very flexible
and allows an unrestricted selection of preferred image
features one to freely choose the preferred image features
to adapt the algorithm to individual needs. An evaluation
on the Berkeley Segmentation Dataset (BSD) shows the
consistency of our approach.

**Keywords:** Image segmentation, region growing, probabilistic
model, naive Bayes, fast kernel density estimation

### Explanation

Semantic image segmentation is an important part of image
processing. To date, different approaches for image segmentation
have been published. However, many of the approaches focus on a single
image feature like texture, color or edges to separate
the image. We propose a region growing algorithm which
combines several cues to grow the regions to a segmented
image.
Our region growing algorithm starts with a set of SLIC1 superpixels and grows the regions until every image pixel
is assigned to just one segment. Our approach uses several
cues, combined in a probabilistic manner, to decide whether
regions should be merged or not. The calculation of the
features of a region are based on basic feature descriptors
to allow a fast computation time. The probability model
assumes independence of the features and therefore multiplies
their likelihoods based on the naive Bayes theorem.

Although our segmentation algorithm didn’t achieve a
high score on the BSD, it is still a useful and flexible tool
with high potential to improve its score on the BSD. An enhancement
could be achieved by substituting our very basic
feature descriptors with more complex and exact descriptors.
A possibility would be the use of Textons or covariance matrices as feature descriptors. Our region growing approach
estimates the merging probability of the adjacent regions
from each region or superpixel and merges every region
which has a higher merging probability then a defined
threshold. This allows a fast computation but merging decisions
for complex regions have to be performed already in
early iteration steps of the region growing. This could be the
reason why we obtain an oversegmentation for complex images.
An idea could be to implement our working approach
with the heap method. For each superpixel, we could have a
heap that stores the regions and their merging probabilities
that can be merged with the superpixel. The root of each
heap would store the region that has the highest merging
probability with the superpixel. Beside the region growing, also the likelihood estimation could be designed differently
to allow an increased flexibility to changes in the training data. At the moment, a separate Matlab script needs to be
run to calculate the correct bandwidth of the KDE kernel.
It would be more convenient, if the algorithm estimated the
correct bandwidth at an initialization step of the program.
