set DSTPATH=../../bin/qtrk
cp cudart32_50_35.dll %DSTPATH%
cp qtrkcudad-labview.dll %DSTPATH%
cp qtrkcudad-labview.pdb %DSTPATH%

cp qtrkcuda-labview.dll %DSTPATH%
cp qtrkcuda-labview.pdb %DSTPATH%

cp qtrk-labview.dll %DSTPATH%
cp qtrk-labview.pdb %DSTPATH%

cp qtrkd-labview.dll %DSTPATH%
cp qtrkd-labview.pdb %DSTPATH%

rm -r %DSTPATH%/QTrkLVBinding
cp -r QTrkLVBinding %DSTPATH%
cp QTrk.mnu %DSTPATH%
pause