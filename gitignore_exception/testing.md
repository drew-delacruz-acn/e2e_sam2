/tmp/ipykernel_56008/619307704.py:8: DeprecationWarning: numpy.core.numeric is deprecated and has been renamed to numpy._core.numeric. The numpy._core namespace contains private NumPy internals and its use is discouraged, as NumPy internals can change without warning in any release. In practice, most real-world usage of numpy.core is to access functionality in the public NumPy API. If that is the case, use the public NumPy API. If not, you are using NumPy internals. If you would still like to access an internal attribute, use numpy._core.numeric._frombuffer.
  resnet_df = pickle.load(file)
resnet_df:                                          video frame             owl_label  \
0  Scenes 001-020__314-3_20230815232058756.mp4   112  Beige collared shirt   
1  Scenes 001-020__314-3_20230815232058756.mp4    58  Beige collared shirt   
2  Scenes 001-020__314-3_20230815232058756.mp4   115  Beige collared shirt   
3  Scenes 001-020__314-3_20230815232058756.mp4    83  Beige collared shirt   
4  Scenes 001-020__314-3_20230815232058756.mp4    52             Dark coat   

                                 finetuned_embedding  
0  [0.0048876973, 0.019972654, 0.048866335, 0.001...  
1  [0.0041305386, 0.026740564, 0.038905628, 0.001...  
2  [0.00619403, 0.01454955, 0.044577274, 0.003630...  
3  [0.009226867, 0.020863937, 0.03188111, 0.00792...  
4  [0.014232857, 0.037429027, 0.03028806, 0.00619...  
def Objects
                                 finetuned_embedding                  class
0  [-0.011280404, 0.049807936, 0.13480993, 0.0647...  Aligator Loki Plushie
1  [0.015214803, 0.031002529, 0.014353202, 0.0263...    Boastful Loki Armor
2  [0.089334205, -0.021669062, 0.01481582, -0.008...     Classic Loki Armor
3  [0.040368285, -0.011553298, -0.0020739324, -0....         Kid Loki Armor
4  [0.036457542, 0.044046428, 0.049783107, -0.018...           Loki's Armor
  visual_predicted_object  visual_max_score
0    TVA Prisoner Uniform          0.657228
1    TVA Prisoner Uniform          0.672014
2    TVA Prisoner Uniform          0.679791
3   Aligator Loki Plushie          0.650488
4          Sylvie's Armor          0.672516
final_SOT:                                                 video  second  \
0         Scenes 001-020__314-3_20230815232058756.mp4      80   
1         Scenes 001-020__314-3_20230815232058756.mp4      80   
2         Scenes 001-020__314-3_20230815232058756.mp4      80   
3         Scenes 001-020__314-3_20230815232058756.mp4      80   
4         Scenes 001-020__314-3_20230815232058756.mp4      80   
...                                               ...     ...   
196551  Scenes 001-020__303L-1-_20230815223024888.mp4      26   
196552  Scenes 001-020__303L-1-_20230815223024888.mp4      26   
196553  Scenes 001-020__303L-1-_20230815223024888.mp4      26   
196554  Scenes 001-020__303L-1-_20230815223024888.mp4      26   
196555  Scenes 001-020__303L-1-_20230815223024888.mp4      26   

                          tag frame  video_id  actual  
0                Loki's Armor  1921       NaN     0.0  
1                  TVA Collar  1921       NaN     0.0  
2                 TVA Uniform  1921       NaN     0.0  
3                  Time Stick  1921       NaN     0.0  
4                      TemPad  1921       NaN     0.0  
...                       ...   ...       ...     ...  
196551              TimeSpear   625       NaN     0.0  
196552    Boastful Loki Armor   625       NaN     0.0  
196553     Classic Loki Armor   625       NaN     0.0  
196554         Kid Loki Armor   625       NaN     0.0  
196555  Aligator Loki Plushie   625       NaN     0.0  

[196452 rows x 6 columns]
TP:  68
FP:  62
FN:  68
TN:  1620
/tmp/ipykernel_56008/619307704.py:55: DeprecationWarning: numpy.core.numeric is deprecated and has been renamed to numpy._core.numeric. The numpy._core namespace contains private NumPy internals and its use is discouraged, as NumPy internals can change without warning in any release. In practice, most real-world usage of numpy.core is to access functionality in the public NumPy API. If that is the case, use the public NumPy API. If not, you are using NumPy internals. If you would still like to access an internal attribute, use numpy._core.numeric._frombuffer.
  final_SOT = pickle.load(file)