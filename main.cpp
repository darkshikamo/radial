#include <iostream>
#include <vector>
#include <cstdlib>
#include <algorithm>    // std::sort
#include <utility>   // std::pair
#include <ctime>
#include <fstream>   // to save a matrix in a text file

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

// g++ -Wall main.cpp -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_calib3d -o radial


// variables globales
const uint nbIterations = 30;
const uint chessboardCorners = 17;
const std::string videoFileName = "input/video_17x17_01.mp4"; //"input/mire_7x7_sanyo_2.mp4";
const double mapScaleRatio = 0.2;

// structures
typedef struct pixelConstraint{
	std::vector< std::pair<uint, uint> > line; // starting and ending points of the line that contains the current point
	cv::Point3d pos;
}pixelConstraint;



//////////////////////////////////////////////////////////////////////////////////
unsigned int getIndex(const unsigned int x, const unsigned int y, const unsigned int imageWidth){
  return y*imageWidth + x;
}

//////////////////////////////////////////////////////////////////////////////////
void addGridPoints(std::vector< pixelConstraint > &nodes, cv::Mat &image, const uint chessboardNbCorners, const double &ratio){

    // detect chessboard
    std::vector<cv::Point2f> imageCorners;                                   // output vectors of image points
    cv::Size boardSize(chessboardNbCorners,chessboardNbCorners);             // number of corners on the chessboard
    bool found = cv::findChessboardCorners(image,boardSize, imageCorners);   // Get the chessboard corners

    // if(!found) for(uint i=0; i<imageCorners.size();++i)
    // 	std::cout << imageCorners[i] << std::endl;

    if(!found) return;

    cv::drawChessboardCorners(image, boardSize, imageCorners,found);         // Draw the corners if found


    // push the nodes
    for(uint i=1; i<chessboardNbCorners-1; ++i)
      for(uint j=1; j<chessboardNbCorners-1; ++j){

      	// the position of the current corner
        uint posx = imageCorners[i*chessboardNbCorners + j].x * ratio;
        uint posy = imageCorners[i*chessboardNbCorners + j].y * ratio;

        // the position of the first and last corner of the same row
        uint firstIndexRow = getIndex(imageCorners[i*chessboardNbCorners].x*ratio,imageCorners[i*chessboardNbCorners].y*ratio,image.size().width*ratio);
        uint lastIndexRow  = getIndex(imageCorners[i*chessboardNbCorners+chessboardNbCorners-1].x*ratio,imageCorners[i*chessboardNbCorners+chessboardNbCorners-1].y*ratio,image.size().width*ratio);

        // the position of the first and last corner of the same colomn
        uint firstIndexCol = getIndex(imageCorners[j].x*ratio,imageCorners[j].y*ratio,image.size().width*ratio);
        uint lastIndexCol  = getIndex(imageCorners[(chessboardNbCorners-1)*chessboardNbCorners + j].x*ratio,imageCorners[(chessboardNbCorners-1)*chessboardNbCorners + j].y*ratio,image.size().width*ratio);

        nodes[getIndex(posx,posy,image.size().width*ratio)].line.push_back( std::pair<uint,uint>(firstIndexRow,lastIndexRow) );
        nodes[getIndex(posx,posy,image.size().width*ratio)].line.push_back( std::pair<uint,uint>(firstIndexCol,lastIndexCol) );
      }


#if 0
     uint topLeftx  =  imageCorners[0].x * ratio;
     uint topLefty  =  imageCorners[0].y * ratio;
     uint topRightx =  imageCorners[chessboardNbCorners-1].x * ratio;
     uint topRighty =  imageCorners[chessboardNbCorners-1].y * ratio;
     uint bottomRightx = imageCorners[(chessboardNbCorners-1)*chessboardNbCorners+chessboardNbCorners-1].x * ratio;
     uint bottomRighty = imageCorners[(chessboardNbCorners-1)*chessboardNbCorners+chessboardNbCorners-1].y * ratio;
     uint bottomLeftx  = imageCorners[(chessboardNbCorners-1)*chessboardNbCorners].x * ratio;
     uint bottomLefty  = imageCorners[(chessboardNbCorners-1)*chessboardNbCorners].y * ratio;

     uint topLeftIndex     = getIndex(topLeftx,topLefty,image.size().width*ratio);
     uint topRightIndex    = getIndex(topRightx,topRighty,image.size().width*ratio);
     uint bottomRightIndex = getIndex(bottomRightx,bottomRighty,image.size().width*ratio);
     uint bottomLeftIndex  = getIndex(bottomLeftx,bottomLefty,image.size().width*ratio);

    // push the frame of the chessboard
    for(uint i=1; i<chessboardNbCorners-1; ++i){

    	// the position of the top / botttom / midlle corners (j=0 | j=nbCcorners | j=nbCorners/2 )
        uint posxBegin = imageCorners[i*chessboardNbCorners].x * ratio;
        uint posyBegin = imageCorners[i*chessboardNbCorners].y * ratio;
        uint posxEnd   = imageCorners[i*chessboardNbCorners+chessboardNbCorners-1].x * ratio;
        uint posyEnd   = imageCorners[i*chessboardNbCorners+chessboardNbCorners-1].y * ratio;
        uint posxMid   = imageCorners[i*chessboardNbCorners+chessboardNbCorners/2].x * ratio;
        uint posyMid   = imageCorners[i*chessboardNbCorners+chessboardNbCorners/2].y * ratio;

        // index of the top / bottom / middle corners
        uint firstIndexRow = getIndex(posxBegin,posyBegin,image.size().width*ratio);
        uint lastIndexRow  = getIndex(posxEnd,posyEnd,image.size().width*ratio);
        uint midleIndexRow = getIndex(posxMid,posyMid,image.size().width*ratio);

        // push first and last
        nodes[firstIndexRow].line.push_back( std::pair<uint,uint>(midleIndexRow,lastIndexRow) );
        nodes[lastIndexRow].line.push_back( std::pair<uint,uint>(firstIndexRow,midleIndexRow) );
        nodes[firstIndexRow].line.push_back( std::pair<uint,uint>(topLeftIndex,bottomLeftIndex) );
        nodes[lastIndexRow].line.push_back( std::pair<uint,uint>(topRightIndex,bottomRightIndex) );



    	// the position of the right / left / midlle corners
    	posxBegin = imageCorners[i].x * ratio;
        posyBegin = imageCorners[i].y * ratio;
        posxEnd   = imageCorners[(chessboardNbCorners-1)*chessboardNbCorners+i].x * ratio;
        posyEnd   = imageCorners[(chessboardNbCorners-1)*chessboardNbCorners+i].y * ratio;
        posxMid   = imageCorners[(chessboardNbCorners/2)*chessboardNbCorners+i].x * ratio;
        posyMid   = imageCorners[(chessboardNbCorners/2)*chessboardNbCorners+i].y * ratio;

        // index of the right / left / middle corners
        firstIndexRow = getIndex(posxBegin,posyBegin,image.size().width*ratio);
        lastIndexRow  = getIndex(posxEnd,posyEnd,image.size().width*ratio);
        midleIndexRow = getIndex(posxMid,posyMid,image.size().width*ratio);

        // push first and last
        nodes[firstIndexRow].line.push_back( std::pair<uint,uint>(midleIndexRow,lastIndexRow) );
        nodes[lastIndexRow].line.push_back( std::pair<uint,uint>(firstIndexRow,midleIndexRow) );
        nodes[firstIndexRow].line.push_back( std::pair<uint,uint>(topLeftIndex,topRightIndex) );
        nodes[lastIndexRow].line.push_back( std::pair<uint,uint>(bottomLeftIndex,bottomRightIndex) );
    }  
#endif

}


//////////////////////////////////////////////////////////////////////////////////
uint findAnkor(const std::vector<pixelConstraint> &nodes, const uint imageWidth, const uint imageHeight){

	const uint centerWidth = imageWidth/2;
	const uint centerHeight = imageHeight/2;

	uint index=0;
    double minDist = imageWidth+imageHeight; // distance bigger than a possible distance
	for(uint i=0; i<nodes.size(); ++i){
    if(nodes[i].line.size()>1){ // if at least 2 lines passing throw this point
      double tmpDist = cv::norm(cv::Point2d(centerWidth,centerHeight)-cv::Point2d(nodes[i].pos.x, nodes[i].pos.y));
      if(tmpDist < minDist){
        index = i;
        minDist = tmpDist;
      }
    }
  }

  return index;
}


//////////////////////////////////////////////////////////////////////////////////
void drawSomeNodes(cv::Mat &img, const std::vector<pixelConstraint> &nodes, const double &ratio){

  img.setTo(cv::Scalar(0));
  for(int i=0; i<10; ++i){
  	int nodeIndex = random() % nodes.size();
  	for(uint j=0;j<nodes[nodeIndex].line.size(); ++j){
  		cv::Point2d pos1,pos2;
  		pos1.x = nodes[nodeIndex].pos.x / ratio;
  		pos1.y = nodes[nodeIndex].pos.y / ratio;
  		pos2.x = nodes[nodes[nodeIndex].line[j].first].pos.x / ratio;
  		pos2.y = nodes[nodes[nodeIndex].line[j].first].pos.y / ratio;

		cv::line(img,
				 pos1, 
				 pos2,
		  		 cv::Scalar(255,255,255));
  		pos2.x = nodes[nodes[nodeIndex].line[j].second].pos.x / ratio;
  		pos2.y = nodes[nodes[nodeIndex].line[j].second].pos.y / ratio;
		cv::line(img,
				 pos1, 
				 pos2,
		  		 cv::Scalar(255,255,255));

		cv::circle(img, pos1, 3, cv::Scalar(0,0,255),-1);
  	}

	if(nodes[nodeIndex].line.size() == 0) i--; // if there was not at least 1 line, this pass does not count
  }
}


//////////////////////////////////////////////////////////////////////////////////
void iterativeSolver(std::vector<pixelConstraint> &nodes, const uint &nbIter, const uint &indexAnkor)
{
 for(uint iter=0; iter<nbIter; ++iter){
    for(uint i=0; i<nodes.size(); ++i) // for each pixel
      if(i!=indexAnkor){
        if(nodes[i].line.size() > 1)  // if more than one constraint
          {
            // build the system
            cv::Mat A(nodes[i].line.size(),2,CV_64F,cv::Scalar(0));
            cv::Mat b(nodes[i].line.size(),1,CV_64F,cv::Scalar(0));
            for(uint n=0; n<nodes[i].line.size(); ++n){
              cv::Point3d a1(nodes[nodes[i].line[n].first].pos);
              cv::Point3d a2(nodes[nodes[i].line[n].second].pos);
              A.at<double>(n,0) = a1.y-a2.y;
              A.at<double>(n,1) = a2.x-a1.x;
              b.at<double>(n)   = a1.y*a2.x-a1.x*a2.y;
            }

            // solve the system and update the pixel position
            cv::Mat res = (A.t()*A).inv(cv::DECOMP_SVD)*A.t()*b;
            nodes[i].pos.x = res.at<double>(0,0);
            nodes[i].pos.y = res.at<double>(1,0);
          }
        }
  }
}

//////////////////////////////////////////////////////////////////////////////////
void fillHoles(cv::Mat &map){

	uint width  = map.size().width;
	uint height = map.size().height;

	for(uint x=1; x<width-1; ++x)
      for(uint y=1; y<height-1; ++y){
        // find a hole
        if(map.at<float>(y,x) == 0){ // this may not be a hole
          // if the 4 neighbours are ok
          if(map.at<float>(y+1,x) != 0 && map.at<float>(y-1,x) != 0 && 
             map.at<float>(y,x+1) != 0 && map.at<float>(y,x-1) != 0){
            map.at<float>(y,x) = 0.25*(map.at<float>(y+1,x)+map.at<float>(y-1,x)+map.at<float>(y,x+1)+map.at<float>(y,x-1)) ;
          }
        }
      }

    for(uint x=1; x<width-1; ++x)
      for(uint y=1; y<height-1; ++y){
        // find a hole
        if(map.at<float>(y,x) == 0){ // this may not be a hole
          // if the 4 neighbours are ok
          if(map.at<float>(y+1,x+1) != 0 && map.at<float>(y-1,x-1) != 0 && 
             map.at<float>(y+1,x-1) != 0 && map.at<float>(y-1,x+1) != 0){
            map.at<float>(y,x) = 0.25*(map.at<float>(y+1,x+1)+map.at<float>(y+1,x-1)+map.at<float>(y-1,x+1)+map.at<float>(y-1,x-1));         }
        }
      }
}

//////////////////////////////////////////////////////////////////////////////////
void saveMatrix(const std::string filename, const cv::Mat A){

    std::ofstream fout(filename.c_str());

    if(!fout){
        std::cerr << "error: could not open file " << filename << std::endl;
        return;
    }
/*
    for(int i=0; i<A.rows; i++){
        for(int j=0; j<A.cols; j++)
            fout << A.at<float>(i,j) << " ";
        fout << std::endl;
    }
*/
    for(int i=0; i<A.rows; i++)
        for(int j=0; j<A.cols; j++)
        	if(A.at<float>(i,j) != 0)
            	fout << i << " " << j << " " << A.at<float>(i,j) << std::endl;

    fout.close();
}

//////////////////////////////////////////////////////////////////////////////////
cv::Mat medianFilter(const cv::Mat &map, const int windowSize){

    cv::Mat final(map.rows,map.cols,CV_32F,cv::Scalar(0.0));

	for(int i=windowSize/2; i<map.rows-windowSize/2; ++i)
		for(int j=windowSize/2; j<map.cols-windowSize/2; ++j){

			// fill the array
			std::vector<float> v;
			for(int k=-windowSize/2; k<windowSize/2; ++k)
				for(int l=-windowSize/2; l<windowSize/2; ++l)
					if(fabs(map.at<float>(i+k,j+l)) > 1.0e-5)
						v.push_back(map.at<float>(i+k,j+l));

			// sort the array
			std::sort(v.begin(), v.end());

			// get the median
			if(v.size()>4)
				final.at<float>(i,j) = v[v.size()/2];
		}

	return final;
}


//////////////////////////////////////////////////////////////////////////////////
cv::Mat gaussianFilter(const cv::Mat &map, const int windowSize){

    cv::Mat final(map.rows,map.cols,CV_32F,cv::Scalar(0.0));

	for(int i=windowSize/2; i<map.rows-windowSize/2; ++i)
		for(int j=windowSize/2; j<map.cols-windowSize/2; ++j){

			// fill the array
			float sum = 0.0;
			int index = 0;
			for(int k=-windowSize/2; k<windowSize/2; ++k)
				for(int l=-windowSize/2; l<windowSize/2; ++l)
					if(fabs(map.at<float>(i+k,j+l)) > 1.0e-5){
						sum += map.at<float>(i+k,j+l);
						index++;
					}

			// get the median
			if(index>2)
				final.at<float>(i,j) = sum /(float)index;
		}

	return final;
}


//////////////////////////////////////////////////////////////////////////////////
int main(void)
{
  // random
  srand(time(0));

  // open video stream
  std::cout << "   open video stream ..." << std::endl;
  cv::VideoCapture capture(videoFileName);
  if(!capture.isOpened()){
    std::cerr << "failed to open video" << std::endl;
    return -1;
  }
  //cap.set(CV_CAP_PROP_FRAME_WIDTH,800);
  //cap.set(CV_CAP_PROP_FRAME_HEIGHT,600);
  double rate= capture.get(CV_CAP_PROP_FPS); 
  int delay= 1000/rate;

  // grab a frame to get the video width and height
  cv::Mat img;
  capture >> img;
  const uint videoWidth  = img.size().width;
  const uint videoHeight = img.size().height;
  std::cout << "      video image width : " << videoWidth << std::endl;
  std::cout << "      video image height: " << videoHeight << std::endl;

  // new size
  const uint mapWidth  = img.size().width * mapScaleRatio;
  const uint mapHeight = img.size().height * mapScaleRatio;
  std::cout << "      map width : " << mapWidth << std::endl;
  std::cout << "      map height: " << mapHeight << std::endl;

  // create and init nodes
  std::cout << "   init nodes ..." << std::endl;
  std::vector< pixelConstraint > nodes(mapWidth * mapHeight);
  for(uint x=0; x<mapWidth; ++x)
    for(uint y=0; y<mapHeight; ++y){
      nodes[getIndex(x,y,mapWidth)].pos.x = x;
      nodes[getIndex(x,y,mapWidth)].pos.y = y;
      nodes[getIndex(x,y,mapWidth)].pos.z = 1.0;
    }

  // image of detected nodes
  cv::Mat nodesImage(mapHeight,mapWidth,CV_8UC3,cv::Scalar(0));

  // display window
  cv::namedWindow("inputVideo");
  cv::namedWindow("dataDisplay");

  // read video stream
  std::cout << "   read video stream ..." << std::endl;
  bool stop = false;
  while(!stop){

    // get the next frame
    if(!capture.read(img)){
    	stop = true;
    	std::cout << "   end of the video stream " << std::endl;
    	break;
    }

    // count frame
    //static uint count=0;
    //std::cout << "      image " << count++ << std::endl;

 
    // push chessboard
    addGridPoints(nodes, img, chessboardCorners, mapScaleRatio);

    // display the detected nodes
    for(uint i=0; i<nodes.size(); ++i)
      if(nodes[i].line.size()>1) nodesImage.at<cv::Vec3b>(nodes[i].pos.y, nodes[i].pos.x)[2] = std::min(nodesImage.at<cv::Vec3b>(nodes[i].pos.y, nodes[i].pos.x)[2]+5,255);

    // display the image
    cv::imshow("inputVideo",img);
    cv::imshow("dataDisplay",nodesImage);

    // adjust the frame rate
    if (cv::waitKey(delay)>=0)
      stop = true;    
  }

  // ankor the center of the image
  std::cout << "   find ankor ... ";
  uint indexAnkor = findAnkor(nodes, mapWidth, mapHeight);
  std::cout << " -> index = " << indexAnkor << std::endl;

  // draw result for some nodes
  std::cout << "   some lines ..." << std::endl;
  img = cv::Mat(videoHeight,videoWidth,CV_8UC3,cv::Scalar(0));
  drawSomeNodes(img,nodes,mapScaleRatio);
  while(true){
  	cv::imshow("inputVideo",img);
  	if(cv::waitKey(10)>0)break;
  };

  // iterative solver
  std::cout << "   iterative solver ..." << std::endl;
  iterativeSolver(nodes, nbIterations, indexAnkor);

  // draw result for some nodes
  img = cv::Mat(videoHeight,videoWidth,CV_8UC3,cv::Scalar(0));
  drawSomeNodes(img, nodes, mapScaleRatio);
  while(true){
  	cv::imshow("inputVideo",img);
  	if(cv::waitKey(10)>0)break;
  };

  // compute a displacement map
  std::cout << "   compute the displacement maps ..." << std::endl;
  cv::Mat displacementMapX(mapHeight,mapWidth,CV_32F,cv::Scalar(0.0));
  cv::Mat displacementMapY(mapHeight,mapWidth,CV_32F,cv::Scalar(0.0));
  for(uint x=0; x<mapWidth; ++x)
    for(uint y=0; y<mapHeight; ++y){

    	// if the considered point was computed
    	if(nodes[getIndex(x,y,mapWidth)].line.size()>1){

	    	// take a point of the final image and have a look at its new position
	    	// here we should interpolate the new position 
	    	cv::Point3d pos = nodes[getIndex(x,y,mapWidth)].pos;

	    	// bound to the image size
	    	pos.x = std::min(pos.x,mapWidth-1.0);
	    	pos.x = std::max(pos.x,0.0);
	    	pos.y = std::min(pos.y,mapHeight-1.0);
	    	pos.y = std::max(pos.y,0.0);

	    	displacementMapX.at<float>(pos.y,pos.x) = x - pos.x;
	    	displacementMapY.at<float>(pos.y,pos.x) = y - pos.y;

	    	// std::cout << "displacementX " << displacementMapX.at<float>(pos.y,pos.x) << std::endl;
	    	// std::cout << "displacementY " << displacementMapY.at<float>(pos.y,pos.x) << std::endl;
	    }
	}

  // denoizing
  std::cout << "   median filter ..." << std::endl;
  displacementMapX = medianFilter(displacementMapX,15);
  displacementMapY = medianFilter(displacementMapY,15);

  std::cout << "   gaussian filter ..." << std::endl;
  displacementMapX = gaussianFilter(displacementMapX,5);
  displacementMapY = gaussianFilter(displacementMapY,5);

  // fill holes
  //std::cout << "   fill the holes ..." << std::endl;
  //fillHoles(displacementMapX);
  //fillHoles(displacementMapY);
  saveMatrix("output/mapX.txt", displacementMapX);
  saveMatrix("output/mapY.txt", displacementMapY);

  // display
  cv::Mat displayMapX,displayMapY;
  displacementMapX.convertTo(displayMapX, CV_8U);
  displacementMapY.convertTo(displayMapY, CV_8U);
  displayMapX *= 50;
  displayMapY *= 50;

  while(true){
  	cv::imshow("inputVideo",displayMapX);
  	cv::imshow("dataDisplay",displayMapY);
  	if(cv::waitKey(10)>0)break;
  };

  // scale displacement map
  std::cout << "   scale displacement map to final size ..." << std::endl;
  displacementMapX = displacementMapX / mapScaleRatio;
  displacementMapY = displacementMapY / mapScaleRatio;
  cv::Mat displacementMapXBig, displacementMapYBig;
  cv::resize(displacementMapX,displacementMapXBig,
  			 cv::Size(displacementMapX.size().width/mapScaleRatio,displacementMapX.size().height/mapScaleRatio),
  			 0,0,cv::INTER_LINEAR);
  cv::resize(displacementMapY,displacementMapYBig,
  			 cv::Size(displacementMapY.size().width/mapScaleRatio,displacementMapY.size().height/mapScaleRatio),
  			 0,0,cv::INTER_LINEAR);
  displacementMapXBig.convertTo(displayMapX, CV_8U);
  displacementMapYBig.convertTo(displayMapY, CV_8U);
  displayMapX *= 20;
  displayMapY *= 20;

  while(true){
  	cv::imshow("inputVideo",displayMapX);
  	cv::imshow("dataDisplay",displayMapY);
  	if(cv::waitKey(10)>0)break;
  };

  // apply the correction to the video
  std::cout << "   rewind the video stream ..." << std::endl;
  capture.set(CV_CAP_PROP_POS_FRAMES,0);  

  std::cout << "   correct the video ..." << std::endl;
  stop = false;
  while(!stop){
  
    // get the next frame
    if(!capture.read(img)){
    	stop = true;
    	std::cout << "   end of the video stream " << std::endl;
    	break;
    }
 
 	// compute the corrected image
 	cv::Mat correctedImage(img.size().height,img.size().width,CV_8UC3,cv::Scalar(0));
	for(int x=0; x<correctedImage.size().width; ++x)
    	for(int y=0; y<correctedImage.size().height; ++y){

    		cv::Point2d pos;
   			pos.x = x + displacementMapXBig.at<float>(y,x);
   			pos.y = y + displacementMapYBig.at<float>(y,x);
    		if(pos.x!=x && pos.y!=y && pos.x!=0 && pos.x!=(correctedImage.size().width-1) && pos.y!=0 && pos.y!=(correctedImage.size().height-1) ) {
    			for(uint c=0; c<3; ++c)
    				correctedImage.at<cv::Vec3b>(y,x)[c] = img.at<cv::Vec3b>((int)pos.y,(int)pos.x)[c];
    		}
    	}

    // display the image
    cv::imshow("dataDisplay",correctedImage);
    cv::imshow("inputVideo",img);

    // adjust the frame rate
    if (cv::waitKey(delay)>=0)
      stop = true;    
  }

  // close the video streaming
  capture.release();

  return 0;
}
