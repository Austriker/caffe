// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include "caffe/fast_rcnn_layers.hpp"

namespace caffe {

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  has_weights_ = (bottom.size() == 3);
}

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  if (has_weights_) {
    CHECK_EQ(bottom[0]->channels(), bottom[2]->channels());
    CHECK_EQ(bottom[0]->height(), bottom[2]->height());
    CHECK_EQ(bottom[0]->width(), bottom[2]->width());
  }
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  errors_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());    // d := b0 - b1
  if (has_weights_) { //to check with weights
    caffe_mul(
        count,
        bottom[2]->cpu_data(),
        diff_.cpu_data(),
        diff_.mutable_cpu_data());  // d := w * (b0 - b1)
  }
	const Dtype* in= diff_.cpu_data();
	Dtype* out= errors_.mutable_cpu_data();
	for(int i=0; i<count; ++i) {
		Dtype val= in[i];
		Dtype abs_val = fabs(val);
    if (abs_val < 1) {
      out[i] = 0.5 * val * val;
    } else {
      out[i] = abs_val - 0.5;
    }
	}
	Dtype loss;
	loss= caffe_cpu_asum(count, errors_.cpu_data());
	top[0]->mutable_cpu_data()[0] = loss / bottom[0]->num();
	
	/*const Dtype* b1= bottom[0]->cpu_data();
	const Dtype* b2= bottom[1]->cpu_data();
	//const Dtype* b1= bottom[0]->.cpu_data();
	std::cout<<"has_weight "<<has_weights_<<"\n";
	int v=std::min(count,10);
	for(int i=0; i<v; ++i) {
		std::cout<<b1[i]<<"\t"<<b2[i]<<"\t"<<in[i]<<"\t"<<out[i]<<"\t"<<"\n";
		std::cout<<fabs(in[i])<<"\n";
	}
	std::cout<<"asum_loss\t"<<loss<<"bottom 0 num\t"<<bottom[0]->num()<<"\n";
	std::cout<<top[0]->mutable_cpu_data()[0]<<"\n";
	*/
 //NOT_IMPLEMENTED;
}

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	int count = diff_.count();
	const Dtype* in= diff_.cpu_data();
	Dtype* out= diff_.mutable_cpu_data();
	for(int i=0; i<count; ++i) {
		Dtype val= in[i];
		Dtype abs_val = fabs(val);
    if (abs_val < 1) {
      out[i] = val;
    } else {
      out[i] = (Dtype(0) < val) - (val < Dtype(0));
    }
	}
	for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha,                           // alpha
          diff_.cpu_data(),                // x
          Dtype(0),                        // beta
          bottom[i]->mutable_cpu_diff());  // y
    }
  }
	//NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(SmoothL1LossLayer);
#endif

INSTANTIATE_CLASS(SmoothL1LossLayer);
REGISTER_LAYER_CLASS(SmoothL1Loss);

}  // namespace caffe
