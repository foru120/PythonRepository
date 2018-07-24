#todo 텐서플로우 설치
'''
    1. Python 2.5 또는 더 높은 버전 설치
    2. Anaconda 설치
     - https://www.continuum.io/downloads
    3. Anaconda 설치 확인
     - conda --version
    4. TensorFlow CPU/GPU 선택
     - TensorFlow GPU 는 CUDA compute capability 3.0 이상이어야 한다.
     - TensorFlow GPU 는 CUDA Toolkit 7.0 이상, cuDNN v3 이상 설치되어 있어야 한다.recipe
     - Window 의 경우 DLL file 들이 추가적으로 필요하다.(DLL file 다운 or Visual Studio C++ 설치)
     - 다른 디렉토리에 설치된 cuDNN file 들은 환경 변수에 디렉토리 경로를 추가하거나 기존 CUDA 설치 디렉토리에 복사한다.
    5. conda 환경 생성
     - conda create -n tensorflow python=3.5
    6. conda 환경 활성화
     # Windows
      - activate tensorflow
     # Mac OS / Ubuntu
      - source activate tensorflow
    7. 텐서플로우 설치
     ## Windows
     # CPU version
      - pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/windows/cpu/tensorflow-1.3.0cr2-cp35-cp35m-win_amd64.whl
     # GPU version
      - pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/windows/gpu/tensorflow_gpu-1.3.0cr2-cp35-cp35m-win_amd64.whl

     ## Mac OS
     # CPU version
      - pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.3.0cr2-py3-none-any.whl
     # GPU version
      - pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/mac/gpu/tensorflow_gpu-1.3.0cr2-py3-none-any.whl

     ## Ubuntu
     # CPU version
      - pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.3.0cr2-cp35-cp35m-linux_x86_64.whl
     # GPU version
      - pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.3.0cr2-cp35-cp35m-linux_x86_64.whl
'''

#todo Jupyter notebook 설치
'''
    1. ipython 설치
     - conda install -c anaconda ipython
    2. nb_conda_kernels 설치
     - conda install -channel=conda-forge nb_conda_kernels
    3. Jupyter notebook 수행
     - jupyter notebook
'''