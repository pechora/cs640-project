# cs640-project

## Abstract
In this project, our goal is to use the prediction of spindles, a pattern of brain waves that occur during non-rapid eye movement sleep, to tackle spindle deficiency in patients suffering from neurological disorders. For this milestone report, we will demonstrate the method to predict sleep spindle patterns at a specific timestamp based on data of readings ahead of this timestamp.

Sleep spindles are generated in the thalamus and are linked to memory consolidation. Recently, by using a combination of thalamic and cortical human recordings, the Bastuji et al show that local spindles are present in the thalamic reticular nucleus but also in the posterior thalamus. Those spindles travel from the thalamus to the cortex and are present in equal amounts, without a specificity in distribution around the cortex, though predominantly detectable on the scalp in the frontal midline region. Based on the data we gathered, (quote on data here) we found out that the longest sleep spindles are shorter than 5 seconds, which means in order to make predictions on potential sleep spindles at timestamp t, we need to derive the predictions of convolution patterns for the next 5-10 seconds after t. Such predictions will be made base on data readings from 3-5 minutes ahead of t.

