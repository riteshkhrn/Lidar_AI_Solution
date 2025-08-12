# remove previous cloned cumm first.
set -e
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

SPCONV_DIR=$SCRIPT_DIR/spconv
LIBSPCONV_DIR=$SPCONV_DIR/example/libspconv

export CUMM_CUDA_VERSION=12.0 # cuda version, required but only used for flag selection when build libspconv.
export CUMM_CUDA_ARCH_LIST="7.5;8.6" # cuda arch flags
export CUMM_DISABLE_JIT=1
export SPCONV_DISABLE_JIT=1

if ! [[ -d $SPCONV_DIR ]]; then
  git clone --branch v2.3.8 https://github.com/traveller59/spconv.git $SPCONV_DIR
else
  echo "SPCONV exist at $SPCONV_DIR"
fi 

if ! [[ -d $LIBSPCONV_DIR/cumm ]]; then
  git clone --branch v0.7.11 https://github.com/FindDefinition/cumm.git $LIBSPCONV_DIR/cumm
  cd $LIBSPCONV_DIR/cumm
  pip install . 
else
  echo "CUMM exist at $LIBSPCONV_DIR/cumm"
fi

cd $SPCONV_DIR
echo "__version__ = '2.3.8'" > spconv/__version__.py # hack so as not to install spconv
python -m spconv.gencode --include=$LIBSPCONV_DIR/spconv/include --src=$LIBSPCONV_DIR/spconv/src --inference_only=True

export CUMM_INCLUDE_PATH="\${CUMM_INCLUDE_PATH}" # if you use cumm as a subdirectory, you need this to find cumm includes.
mkdir -p $LIBSPCONV_DIR/build
cd $LIBSPCONV_DIR/build
cmake -DCMAKE_CXX_FLAGS="-DTV_CUDA" ..
cmake --build $LIBSPCONV_DIR/build --config Release -j 8 # --verbose
