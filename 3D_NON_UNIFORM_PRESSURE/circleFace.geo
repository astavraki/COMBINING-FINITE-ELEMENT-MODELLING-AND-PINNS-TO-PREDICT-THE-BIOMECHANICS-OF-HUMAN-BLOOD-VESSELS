Mesh.RecombineAll = 0;
Mesh.RecombinationAlgorithm = 0;
SetFactory("OpenCASCADE");
Merge "circleFace.step";
//+
Transfinite Curve {1, 2} = 35 Using Progression 1;
//+
Extrude {0, 0, 20} {
  Surface{1}; Layers {35}; 
}
