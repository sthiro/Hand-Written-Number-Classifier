# HandWrittenNumberClassifier

![image](https://github.com/sthiro/HandWrittenNumberClassifier/assets/49124307/57d3de63-4d6d-4b0c-9555-42dd7036f5cf)

<h2> What Does it do ?</h2>
<p>It captures live video from the computer's webcam and predicts the number using CNN (Convolutional neural network)</p>
<p>Here Pytorch and OpenCV are the main packages used. Pytorch is used to make CNN Model and OpenCV is used to capture the Video</p>

<h2>How Does it do ?</h2>
<ol><li> The video frames is changed to 28 x 28 pixels</ol>
    <li> The video frames are converted into  Grayscale image</li>
    <li>The Grayscale image is converted to Binary Image</li>
    <li>And some pre-image processing at the end (Flooding)</li>
    <li>Finally its feeds into CNN Model and predicts it</li>
</ol>

<h2> Basic Overview of CNN Model</h2>

![representing paper image](https://github.com/sthiro/HandWrittenNumberClassifier/assets/49124307/778dda8b-c1cf-4c58-a7ab-2e6b4f3aebb1)

<h1>Shoud I need to train ?</h1>
<p>No need to train again, The model has trained so well already, a trained file is given there.</p>

<h4> By - S.Thiroshan</h4>
