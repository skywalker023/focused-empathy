<!--# Pragmatic Self-Consciousness<br>for Improving Persona Consistency in Dialogues-->
# Perspective-taking and Pragmatics for Generating<br>Empathetic Responses Focused on Emotion Causes

<!--![figure](assets/figure1.png)-->

**Official PyTorch implementation and EmoCause evaluation set of our EMNLP 2021 paper 💛:**<br>
[Hyunwoo Kim](https://hyunw.kim), [Byeongchang Kim](https://bckim92.github.io), and [Gunhee Kim](https://vision.snu.ac.kr/gunhee). Perspective-taking and Pragmatics for Generating Empathetic Responses Focused on Emotion Causes. _EMNLP_, 2021 [[Paper coming soon!]]()

<!--* **TL;DR**: Inspired by social cognition and pragmatics, we model _public self-consciousness_ in existing dialogue agents with an imaginary listener to improve consistency. Compared to previous works, our method does not require additional consistency-related labels or training.-->

<!--Earlier version of this work was also accepted at ICLR 2020 [Bridging AI and Cognitive Science (BAICS) workshop](https://baicsworkshop.github.io/) as an oral presentation.-->


<!--## Reference-->

<!--If you use the materials in this repository as part of any published research, we ask you to cite the following [paper](https://arxiv.org/abs/2004.05816):-->

<!--```bibtex-->
<!--@inproceedings{Kim:2020:selfc,-->
  <!--title={Will I Sound Like Me? Improving Persona Consistency in Dialogues through Pragmatic Self-Consciousness},-->
  <!--author={Kim, Hyunwoo and Kim, Byeongchang and Kim, Gunhee},-->
  <!--booktitle={EMNLP},-->
  <!--year=2020-->
<!--}-->
<!--```-->

### Have any question?
Please contact [Hyunwoo Kim](https://hyunw.kim) at hyunw.kim@vl.snu.ac.kr.

<!--## Implementation-->

<!--### System Requirements-->

<!--* Python 3.7.9-->
<!--* Pytorch 1.6.0-->
<!--* CUDA 10.2 supported GPU with at least 24GB memory-->
<!--* See [environment.yml](https://github.com/skywalker023/focused-empathy/blob/master/environment.yml) for details-->

<!--### Environment setup-->

<!--Our code is built on the [ParlAI](https://parl.ai/) framework.<br>-->
<!--We recommend you create a conda environment as follows-->

<!--```bash-->
<!--conda env create -f environment.yml-->
<!--```-->

<!--and activate it with-->

<!--```bash-->
<!--conda activate focused-empathy-->
<!--```-->
## EmoCause Evaluation set

Coming soon!

## Running Experiments

Coming soon!

<!--### Self-conscious Blender for its persona-->

<!--#### Dialogue NLI-->

<!--```bash-->
<!--python eval_dnli.py --conscious-target self -t tasks.teachers:SelfConsciousDialogueTeacher --model agents.selfconscious_blender:SelfConsciousBlenderAgent --fp16 false-->
<!--```-->

<!--#### PersonaChat-->

<!--```bash-->
<!--python eval_personachat.py --conscious-target self -t tasks.teachers:SelfConsciousDialogueTeacher --model agents.selfconscious_blender:SelfConsciousBlenderAgent --batchsize 48 --fp16 false-->
<!--```-->

<!--### Self-conscious Blender for its context-->

<!--#### Dialogue NLI-->

<!--```bash-->
<!--python eval_dnli.py --conscious-target context -t tasks.teachers:ContextConsciousDialogueTeacher --model agents.selfconscious_blender:SelfConsciousBlenderAgent --fp16 false-->
<!--```-->

<!--#### PersonaChat-->

<!--```bash-->
<!--python eval_personachat.py --conscious-target context -t tasks.teachers:ContextConsciousDialogueTeacher --model agents.selfconscious_blender:SelfConsciousBlenderAgent --batchsize 48 --fp16 false-->
<!--```-->

<!--💡 In case you want to run the evaluation with vanilla Blender as is, set the `--conscious-target` to `none`.-->


<!--## Acknowledgements-->

<!--We would like to thank [Reuben Cohn-Gordon](https://reubencohngordon.com/), [Sean Welleck](https://cs.nyu.edu/~welleck/), [Junhyug Noh](https://junhyug.github.io/) and [Jiwan Chung](https://vl.snu.ac.kr/people/jiwanchung.html) for their valuable comments. -->
<!--We thank the anonymous reviewers for their thoughtful suggestions on this work.-->

<!--This research was supported by Brain Research Program by National Research Foundation of Korea (NRF) (2017M3C7A1047860), Institute of Information \& communications Technology Planning \& Evaluation (IITP) grant funded by the Korea government (MSIT) (No. 2017-0-01772, Video Turing Test, No. 2019-0-01082, SW StarLab), and Creative Pioneering Researchers Program through Seoul National University.-->


## License

This repository is MIT licensed. See the [LICENSE](https://github.com/skywalker023/focused-empathy/blob/master/LICENSE) file for details.
