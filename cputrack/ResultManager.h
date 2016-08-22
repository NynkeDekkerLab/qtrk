#pragma once

#include "QueuedTracker.h"
#include <list>
#include "threads.h"

/** \defgroup RM Result Manager
\brief Module that handles data gathering and saving for QueuedTracker instances.
*/

/** \addtogroup RM
	@{
*/

/*! \brief \b Placeholder. Abstract template for an output file.

Used to generalize output to different types of datafiles. Implemented by ::TextResultFile and ::BinaryResultFile.
\todo Actually make the implementation.
\todo Rewrite resultmanager to use these classes.
*/
class ResultFile
{
public:
	ResultFile() { }
	virtual ~ResultFile() {}
	virtual void LoadRow(std::vector<vector3f>& pos) = 0; 
	virtual void SaveRow(std::vector<vector3f>& pos) = 0;
};

/*! \brief \b Placeholder. Handler for a text output file. Currently empty shell.

*/
class TextResultFile : public ResultFile
{
public:
	TextResultFile(const char* fn, bool write);
	void LoadRow(std::vector<vector3f>& pos);
	void SaveRow(std::vector<vector3f>& pos);
private:
	FILE *f;
};

/*! \brief \b Placeholder. Handler for a binary output file. Currently empty shell.

*/
class BinaryResultFile : public ResultFile
{
public:
	BinaryResultFile(const char* fn, bool write);
	void LoadRow(std::vector<vector3f>& pos);
	void SaveRow(std::vector<vector3f>& pos);
protected:
	FILE *f;
};

/*! \brief Structure for settings used by ::ResultManager. 

Compiled without padding to line up with LabVIEW alignment.
\warning Changing this requires changing of the linked LabVIEW cluster QTrkSettings.ctl.
*/
#pragma pack(push,1)
struct ResultManagerConfig
{
	int numBeads;				///< Number of beads for which to grab results. Should \a always equal the amount of beads in a single frame.
	int numFrameInfoColumns;	///< Number of columns in the frame info metadata file. Additional columns can be added to save more data on a per-frame basis.
	/*! \brief Scaling factor for each of the three dimensions. 
	
	Used to calculate from pixel/plane output from QueuedTracker to physical values. 
	Output will be (position + \ref offset) * \ref scaling.
	*/
	vector3f scaling;

	/*! \brief Offset value for each of the three dimensions. 
	
	Used to calculate from pixel/plane output from QueuedTracker to physical values. 
	Output will be (position + \ref offset) * \ref scaling.
	*/
	vector3f offset;			
	int writeInterval;			///< Interval of number of gathered frames at which to write the data.
	uint maxFramesInMemory;		///< Number of frames for which to keep the data in memory. 0 for infinite.
	uint8_t binaryOutput;		///< Flag (boolean) to output a binary file instead of a text file.
};
#pragma pack(pop)

/*! \brief  Class that handles data gathering and saving from QueuedTracker instances.

Creates a separate thread to check a linked QTrk instance for new data. Gathers data and saves to the disk at regular intervals.
Mutexes used to ensure thread safety. <br>
Results from QueuedTracker are sorted as [frame][bead] to enable saving data on a frame-per-line basis.
*/
class ResultManager
{
public:
	/*! \brief Create an instance of ResultManager.

	\param [in] outfile String (char array) with full path and filename of desired output file for data.
	\param [in] frameinfo String (char array) with full path and filename of desired output file for frame metadata.
	\param [in] cfg Pointer to structure ::ResultManagerConfig with settings to use.
	\param [in] colnames 
	\parblock
	Vector of names for the columns in the frame info file. Size must equal ResultManagerConfig::numFrameInfoColumns. <br>
	Only used for binary files.
	\endparblock
	*/
	ResultManager(const char *outfile, const char *frameinfo, ResultManagerConfig *cfg, std::vector<std::string> colnames);

	/*! \brief Destroy an instance of ResultManager.

	Sets a flag to stop the thread and frees all results from memory.
	*/
	~ResultManager();

	/*! \brief Save an interval of frames to a file.

	\warning Unimplemented.

	\param [in] start Framenumber of the start of the interval to save.
	\param [in] end Framenumber of the end of the interval to save.
	\param [in] beadposfile String (char array) with full path and filename of desired output file for data.
	\param [in] infofile String (char array) with full path and filename of desired output file for frame metadata.
	*/
	void SaveSection(int start, int end, const char *beadposfile, const char *infofile);

	/*! \brief Set the tracker from which to fetch results.

	\param [in] qtrk Pointer to the QueuedTracker instance to fetch results from.
	*/
	void SetTracker(QueuedTracker *qtrk);
	/*! \brief Get the tracker from which results are fetched.

	\return Pointer to the current QueuedTracker instance.
	*/
	QueuedTracker* GetTracker();

	/*! \brief Get the positions of a single bead over an interval of frames.

	\param [in] startFrame Framenumber of the start of the interval to save.
	\param [in] endFrame Framenumber of the end of the interval to save.
	\param [in] bead Beadnumber of the bead for which the results are requested.
	\param [in,out] r 
	\parblock Array of LocalizationResult in which to store the data. <br>
	Has to be initialized before calling this function. Size has to be equal to \p endFrame - \p startFrame.
	\endparblock
	*/
	int GetBeadPositions(int startFrame, int endFrame, int bead, LocalizationResult* r);

	/*! \brief Get the positions of all beads over an interval of frames.

	\param [in] startFrame Framenumber of the start of the interval to save.
	\param [in] numResults Number of frames to save.
	\param [in,out] results
	\parblock Array of LocalizationResult in which to store the data. <br>
	Has to be initialized before calling this function. Size has to be equal to \p numResults.
	\endparblock
	*/
	int GetResults(LocalizationResult* results, int startFrame, int numResults);

	/*! \brief Write all available data regardless of ::ResultManagerConfig::writeInterval.

	Use to write unsaved frames from memory when no new ones are expected anymore. Writes all leftover data after waiting for the linked QueuedTracker
	to finish its queued localizations.
	*/
	void Flush();

	/*! \brief Structure to keep track of frame counts. */
	struct FrameCounters {
		FrameCounters();
		int startFrame;			///< Index of the first frame for which results are available.
		int processedFrames;	///< Number of frames processed by the ResultManager.
		int lastSaveFrame;		///< Index of the last frame saved to the output file.
		int capturedFrames;		///< Number of frames captured. Counted through calls to \ref StoreFrameInfo.
		int localizationsDone;	///< Amount of localizations finished by the linked QueuedTracker instance.
		int lostFrames;			///< Number of frames deleted from memory before results were finished and/or saved.
		int fileError;			///< Count of reported errors while handling output files. Currently only counts on error when opening binary files.
	};

	/*! \brief Returns a FrameCounters structure with the current counts. */
	FrameCounters GetFrameCounters();

	/*! \brief Store metadata for a frame. This data will be saved in the info file.
	
	\param [in] frame Frame number for which to store the data.
	\param [in] timestamp Timestamp for the frame.
	\param [in] columns Array of float values for the metadata columns. Size must equal ResultManagerConfig::numFrameInfoColumns.
	*/
	void StoreFrameInfo(int frame, double timestamp, float* columns);

	/*! \brief Returns the number of captured frames.

	\return Number of captured frames 
	*/
	int GetFrameCount();

	/*! \brief Remove all results of a certain bead.

	Currently only removes from memory.

	\param [in] bead The index of the bead for which to remove the data.
	*/
	bool RemoveBeadResults(int bead);
	
	/*! \brief Returns a reference to the used configuration to review.

	\return The ResultManagerConfig structure used for this instance of ResultManager. Read-only.
	*/
	const ResultManagerConfig& Config() { return config; }

protected:
	/*! \brief Checks if a frame index is valid.

	Frame index is valid if within bounds of the current frames allowed in memory. Initializes the required \ref FrameResult
	to store data for the frame \p fr if it is valid.

	\param [in] fr Frame index to be checked.
	\retval True Index is fine. New \ref FrameResult initialized and added to results vector.
	\retval False Index is invalid. Can't process this frame index.
	*/
	bool CheckResultSpace(int fr);

	/*! \brief Write available results to output files.

	Chooses between \ref WriteBinaryResults and \ref WriteTextResults depending on settings.
	*/
	void Write();

	/*! \brief Write available data to a binary file.

	\warning Does not update frame counters. Use \ref Write and proper settings instead.
	*/
	void WriteBinaryResults();

	/*! \brief Write available data to a text file.

	\warning Does not update frame counters. Use \ref Write and proper settings instead.
	*/
	void WriteTextResults();

	/*! \brief Copies results from QueuedTracker to internal data structures.

	Adds a result to the pre-allocated memory of frame results so it can later be saved on a per-frame basis.
	
	\note Results are changed from pixel to physical values here, using input given on initialization through \ref ResultManagerConfig.
	\param [in] r A pointer to a \ref LocalizationResult of a single localization.
	*/
	void StoreResult(LocalizationResult* r);

	/*! \brief The base loop of the gathering and saving thread. 
	
	Called when initialization is done.
	
	\param [in] param Pointer to the instance of ResultManager to work for.
	*/
	static void ThreadLoop(void *param);

	/*! \brief General worker function called from \ref ThreadLoop.

	Fetches results from QueuedTracker (\ref FetchResults), sorts them (\ref StoreResult) and saves them if needed (\ref Write).
	*/
	bool Update();

	/*! \brief Writes metadata to the header of a binary data file.

	The saved header for file version 3 is:
	<table>
	<tr><th>Name				<th>Description										<th>Size (bytes)
	<tr><td>version				<td>Binary file version								<td>4 (int)
	<tr><td>numBead				<td>Number of beads in the file						<td>4 (int)
	<tr><td>numFrameInfoColumns	<td>Number of metadata columns						<td>4 (int)
	<tr><td>data_offset			<td>File offset to starting point of frame data		<td>8 (long)
	<tr><td>frameInfoNames		<td>The names/identifiers for the metadata columns	<td>variable (\a numFrameInfoColumns * strings)
	</table>
	*/
	void WriteBinaryFileHeader();

	/*! \brief Structure to save all bead results of a single frame in memory. */
	struct FrameResult
	{
		/*! \brief Initialize a new frame. 
		
		\param [in] nResult Number of beads in the frame. Used to initialize \ref results vector.
		\param [in] nFrameInfo Number of frame info columns. Used to initialize \ref frameInfo vector.
		*/
		FrameResult(int nResult, int nFrameInfo) : frameInfo(nFrameInfo), results(nResult) { count=0; timestamp=0; hasFrameInfo=false;}
		std::vector<LocalizationResult> results;	///< Vector of \ref LocalizationResult. To be filled with tracking results.
		std::vector<float> frameInfo;				///< Vector of frame info values. To be filled through \ref StoreFrameInfo.
		int count;									///< Number of completed localizations for this frame.
		double timestamp;							///< Timestamp at which the frame was taken. Set through \ref StoreFrameInfo.
		bool hasFrameInfo;							///< Flag (boolean) indicating whether frame info columns exist.
	};

	Threads::Mutex resultMutex;		///< Mutex to govern access to \ref frameResults.
	Threads::Mutex trackerMutex;	///< Mutex to govern access to the linked QueuedTracker instance.

	std::vector<std::string> frameInfoNames;	///< Vector with frame info column names. See \ref WriteBinaryFileHeader.

	std::deque< FrameResult* > frameResults;	///< Vector to hold the results for all frames.
	FrameCounters cnt;							///< Local instance of \ref FrameCounters to maintain counts accross functions.
	ResultManagerConfig config;					///< Local instance of \ref ResultManagerConfig with the settings used by this instance of \ref ResultManager.

	ResultFile* resultFile;						///< Local instance of \ref ResultFile to generalize output to text or binary files. Not used.

	QueuedTracker* qtrk;						///< Pointer to the QueuedTracker instance to check.

	std::string outputFile;						///< Path and filename of the main data output file.
	std::string frameInfoFile;					///< Path and filename of the frame info (metadata) file.
	Threads::Handle* thread;					///< Handle to the thread running the \ref ThreadLoop.
	Atomic<bool> quit;							///< Flag to exit the threads.
};

/** @} */