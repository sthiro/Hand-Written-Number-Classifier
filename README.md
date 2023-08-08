# HandWrittenNumberClassifier

![image](https://github.com/sthiro/HandWrittenNumberClassifier/assets/49124307/57d3de63-4d6d-4b0c-9555-42dd7036f5cf)

<h2> What Does it do ?</h2>
<p>It captures live video from the computer's webcam and predicts the number using CNN (Convolutional neural network)</p>
<p>Here Pytorch and OpenCV are the main packages used. Pytorch is used to make CNN Model and OpenCV is used to capture the Video</p>

<h2>How Does it do ?</h2>
<ul><li>First the the video frames is changed to 28 x 28 pixels</li>
    <li>Then it's changes into Grayscale</li>
    <li>Then it's converts the Grayscale image into Binary Image</li>
    <li>And some pre-image processing at the end (Flooding)</li>
    <li>Then its feed into CNN Model</li>
</ul>

<h2> Basic Overview of Model</h2>

![representing paper image](https://github.com/sthiro/HandWrittenNumberClassifier/assets/49124307/778dda8b-c1cf-4c58-a7ab-2e6b4f3aebb1)

<h1>Shoud I need to train ?</h1>
<p>No need to train again, The model is trained so well already which is given there.</h1>

<h4> By - S.Thiroshan</h4>
