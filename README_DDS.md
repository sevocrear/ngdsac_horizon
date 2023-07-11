# How to run the node:
```
python3 demo_dds.py
``````
## Usage: 

**Options:**

  **--topic_in_img** TEXT             name of the input image topic

  **--topic_out_debug_img** TEXT      name of the output debug image topic

  **--topic_out_hor_coeff** TEXT      name of the horizon Line  coeffs topic

  **-m, --model** TEXT                a trained network

  **-c, --capacity** INTEGER          controls the model capactiy of
                                  the network, must match the
                                  model to load (multiplicative
                                  factor for number of channels)

  **-is, --imagesize** INTEGER        size of input images to the
                                  network, must match the model
                                  to load

  **-it, --inlier_thr** FLOAT         threshold used in the soft
                                  inlier count, relative to
                                  image size

  **-ia, --inlier_alpha** FLOAT       scaling factor for the soft
                                  inlier scores (controls the
                                  peakiness of the hypothesis
                                  distribution)

  **-ib, --inlier_beta** FLOAT        scaling factor within the
                                  sigmoid of the soft inlier
                                  count

  **-hyps, --hypotheses** INTEGER     number of line hypotheses
                                  sampled for each image

  **-u, --uniform** TEXT              disable neural-guidance and
                                  sample data points uniformely,
                                  use with a DSAC model

  **-st, --score_thr** FLOAT          threshold on soft inlier count
                                  for drawing the estimate
                                  (range 0-1)

  **-d, --device** TEXT               Device to run the neural
                                  network (cuda, cpu)

  **--record / --no-record** --record video or not

  **--show / --no-show** - show frame or not

  **--plot / --no-plot** - plot data or not

  --**smooth-type** [None|Kalman|Particles]

  --**help**                          Show this message and exit.