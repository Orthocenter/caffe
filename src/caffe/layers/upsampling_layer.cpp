#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "opencv2/opencv.hpp"

namespace caffe {

template <typename Dtype>
void UpsamplingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	vector<Blob<Dtype>*>* top)
{
	UpsamplingParameter upsample_param = this->layer_param_.upsampling_param();
	
	CHECK_GT(upsample_param.new_width(), 0) << "Width cannot be zero.";
	CHECK_GT(upsample_param.new_height(), 0) << "Height cannot be zero.";
	
	newWidth = upsample_param.new_width();
	newHeight = upsample_param.new_height();
	
	areaRatio = (double)upsample_param.new_width() * upsample_param.new_height() /
		bottom[0]->width() / bottom[0]->height();
}

template <typename Dtype>
void UpsamplingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	vector<Blob<Dtype>*>* top)
{
	num = bottom[0]->num();
	channels = bottom[0]->channels();
	width = bottom[0]->width();
	height = bottom[0]->height();

	(*top)[0]->Reshape(num, channels, newHeight, newWidth);
}

template <typename Dtype>
void UpsamplingLayer<Dtype>::Forward_cpu(
	const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top)
{
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = (*top)[0]->mutable_cpu_data();
	
	for(int i = 0; i < num; i++)
	{
		// each channel
		for(int j = 0; j < channels; j++)
		{
			// copy into cv::Mat
			cv::Mat img(height, width, CV_64F);
			
			int nr = height, nc = width;
			
			// if img is cotinuous then regard it as one-row vector
			if(img.isContinuous())
			{
				nr = 1;
				nc = height * width;
			}
			
			// each row
			for(int r = 0; r < nr; r++)
			{
				double *img_data = img.ptr<double>(r);
				
				// each column
				for(int c = 0; c < nc; c++)
				{
					*img_data++ = static_cast<double>(*bottom_data++);
				}
			}
			
			cv::resize(img, img, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_AREA);
			
			// copy resized image into top_data
			int nnr = newHeight, nnc = newWidth;
			
			if(img.isContinuous())
			{
				nnr = 1;
				nnc = newHeight * newWidth;
			}
			
			// each row
			for(int r = 0; r < nnr; r++)
			{
				double *img_data = img.ptr<double>(r);
				// each column
				for(int c = 0; c < nnc; c++)
				{	
					*top_data++ = static_cast<Dtype>(*img_data++);
				}
			}
		}
	}
}

template <typename Dtype>
void UpsamplingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom)
{	
	if(propagate_down[0])
	{
		const Dtype *top_diff = top[0]->cpu_diff();
		Dtype *bottom_diff = (*bottom)[0]->mutable_cpu_diff();
		
		for(int i = 0; i < num; i++)
		{
			// each channel
			for(int j = 0; j < channels; j++)
			{
				cv::Mat mat_diff(newHeight, newWidth, CV_64F);
				
				// copy top_diff into mat_diff
				int nr = newHeight, nc = newWidth;
				if(mat_diff.isContinuous())
				{
					nr = 1;
					nc = newHeight * newWidth;
				}
				
				for(int r = 0; r < nr; r++)
				{
					double *mat_diff_data = mat_diff.ptr<double>(r);
					for(int c = 0; c < nc; c++)
					{
						*mat_diff_data++ = *top_diff++;
					}
				}
				
				cv::resize(mat_diff, mat_diff, cv::Size(width, height), 0, 0, cv::INTER_AREA);
				
				// copy resized mat_diff into bottom_diff
				int nnr = height, nnc = width;
				
				if(mat_diff.isContinuous())
				{
					nnr = 1;
					nnc = height * width;
				}
				
				for(int r = 0; r < nnr; r++)
				{
					double *mat_diff_data = mat_diff.ptr<double>(r);
					for(int c = 0; c < nnc; c++)
					{
						*bottom_diff++ = (*mat_diff_data++) * areaRatio;
					}
				}
			}
		}
	}
}


INSTANTIATE_CLASS(UpsamplingLayer);
} // namespace caffe
