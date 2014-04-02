using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using System.IO;
using System.Windows.Forms.DataVisualization.Charting;
using System.Globalization;
using System.Diagnostics;
using System.Threading.Tasks;

namespace TraceViewer
{
	public partial class ViewTraceDlg : Form, IDisposable
	{
		int numFrames;
		int numBeads;
		string[] infoColNames;
		int startOffset;
		int bytesPerFrame;
		bool haveErrors;
		string filename;

		public class Trace
		{
			public List<Frame> frames;
			public int avg;
		}

		public struct BeadInfo
		{
			public float noise, min, max;
		}

		BeadInfo[] beadInfo;
		List<Trace> traces = new List<Trace>();

		public ViewTraceDlg()
		{
			InitializeComponent();
		}

		private void readBinFileToolStripMenuItem_Click(object sender, EventArgs e)
		{

		}

		private void menuOpenFile(object sender, EventArgs e)
		{
			OpenFileDialog ofd = new OpenFileDialog();
			if (ofd.ShowDialog() == System.Windows.Forms.DialogResult.OK)
			{
				filename = ofd.FileName;

				ReadHeader(false);
				UpdateGraph();
			}
		}


		void ResetBeadList()
		{
			beadListView.Items.Clear();
			int refBead;
			if (!int.TryParse(txtRefBead.Text, out refBead))
				refBead = -1;
			
			beadInfo=new BeadInfo[numBeads];
			Parallel.For(0, numBeads, delegate(int i) {
				var data=GetBeadData(traces[0], i, refBead);
				float[] z=Array.ConvertAll(data, v => v.z);
				if (i != refBead)
				{
					float[] smoothed;
					beadInfo[i].noise = Noise(z, 10, out smoothed);
					beadInfo[i].min = smoothed.Min();
					beadInfo[i].max = smoothed.Max();
				}
			});


			for (int i = 0; i < numBeads; i++)
			{
				var lvi=beadListView.Items.Add(new ListViewItem()
				{
					Text = i.ToString(),
					Tag = i
				});

				lvi.SubItems.Add(beadInfo[i].noise.ToString());
				lvi.SubItems.Add(beadInfo[i].min.ToString());
				lvi.SubItems.Add(beadInfo[i].max.ToString());

			}

			
		}

		void ReadHeader(bool oldVersion)
		{
			traces = new List<Trace>();
			Trace tr = new Trace();
			tr.frames = new List<Frame>();
			traces.Add(tr);

			using (FileStream stream = File.OpenRead(filename))
			{
				stream.Seek(0, SeekOrigin.Begin);

				using (BinaryReader r = new BinaryReader(stream))
				{
					int a = r.ReadInt32();
					int b = r.ReadInt32();
					int c = r.ReadInt32();

					int infoCols;
					if (oldVersion)
					{
						numBeads = a;
						infoCols = b;
						startOffset = c;
					}
					else
					{
						int d = r.ReadInt32();
						infoCols = c;

						if (d == 1234)
						{
							// shitty format
							infoCols = 7;
							startOffset = (int)c;
						}
						else startOffset = (int)d;

						numBeads = b; 
						
					}
					infoColNames = new string[infoCols];
					for (int j = 0; j < infoCols; j++)
					{
						infoColNames[j] = ReadZStr(r);
						Console.WriteLine("InfoCol[{0}]={1}", j, infoColNames[j]);
					}
					haveErrors = !oldVersion;
					bytesPerFrame = 4 * 3 * numBeads + infoCols * 4 + 4 + 8 + (haveErrors ? (4 * numBeads) : 0);
					numFrames = ((int)stream.Length - startOffset) / bytesPerFrame;
					Console.WriteLine("#Frames: {0}", numFrames);

					stream.Seek(startOffset, SeekOrigin.Begin);


					for (int i = 0; i < numFrames; i++)
					{
						tr.frames.Add(ReadFrame(r));
					}

					tr.avg = 1;
					ComputeAverages();

					trackBar.Maximum = numFrames;
					trackBar.Value = 0;
				}
			}

			ResetBeadList();
		}

		public struct Vec3 {
			public float x,y,z;
		}

		public class Frame
		{
			public int id;
			public double timestamp;
			public float[] frameInfo;
			public Vec3[] positions;
			public int[] errors;
		}

		void ComputeAverages()
		{
			int i = 1;
			while (true)
			{
				Trace org = traces.LastOrDefault();
				Trace nt = new Trace();
				nt.avg = i*2;
				nt.frames = new List<Frame>(org.frames.Count / 2);

				for (int j = 0; j < org.frames.Count-1; j+=2)
				{
					Frame a=org.frames[j];
					Frame b=org.frames[j+1];

					Frame fr = new Frame();
					fr.positions = new Vec3[a.positions.Length];
					for (int k = 0; k < fr.positions.Length; k++) /*
						fr.positions[k] = new Vec3()
						{
							x = 0.5f * (a.positions[k].x + b.positions[k].x),
							y = 0.5f * (a.positions[k].y + b.positions[k].y),
							z = 0.5f * (a.positions[k].z + b.positions[k].z)
						};*/
						fr.positions[k] = a.positions[k];

					fr.frameInfo = new float[a.frameInfo.Length];
					for (int k = 0; k < a.frameInfo.Length; k++)
						fr.frameInfo[k] = a.frameInfo[k];

					nt.frames.Add(fr);
				}

				traces.Add(nt);
				i *= 2;

				if (nt.frames.Count < 1000)
					break;
			}
		}

		Frame ReadFrame(BinaryReader r)
		{
			Frame f = new Frame();
			f.frameInfo = new float[infoColNames.Length];
			f.positions = new Vec3[numBeads];
			f.id = r.ReadInt32();
			f.timestamp = r.ReadDouble();
			for (int i = 0; i < infoColNames.Length; i++)
				f.frameInfo[i] = r.ReadSingle();
			for (int i = 0; i < numBeads; i++)
			{
				Vec3 v = new Vec3() {
					x=r.ReadSingle(),
					y=r.ReadSingle(),
					z=r.ReadSingle()
				};
				f.positions[i] = v;
			}
			if (haveErrors)
			{
				f.errors = new int[numBeads];
				for (int i = 0; i < numBeads; i++)
					f.errors[i] = r.ReadInt32();
			}
			return f;
		}

		string ReadZStr(BinaryReader r)
		{
			StringBuilder sb = new StringBuilder();
			while (true)
			{
				char b = r.ReadChar();
				if (b == 0)
					return sb.ToString();
				sb.Append(b);
			}
		}

		void UpdateGraph()
		{
			if (traces.Count == 0 || beadListView.SelectedIndices.Count==0)
				return;
			int[] beads = new int[] { beadListView.SelectedIndices[0] };

			int nf = int.Parse(textNumFramesInView.Text);

			float framesPerPixel = nf/(float)chart.Width ;
			Trace tr = traces.FirstOrDefault(t => t.avg*2>= framesPerPixel);
			if (tr == null) tr = traces[0];

			int maxFrame = tr.frames.Count;
			int nFrame = nf / tr.avg;
			int startFrame = Math.Min(trackBar.Value / tr.avg, maxFrame - nFrame);
			if (startFrame < 0) startFrame = 0;
			if (nFrame + startFrame > maxFrame) nFrame=maxFrame-startFrame;
			//Console.WriteLine("StartFrame:{0} ", startFrame);
			
			Frame[] data = tr.frames.GetRange(startFrame, nFrame).ToArray();
			chart.Series.Clear();
			chart.SuspendLayout();

			int refBead = -1;
			if (checkUseRef.Checked && txtRefBead.Text.Length > 0)
			{
				refBead = int.Parse(txtRefBead.Text);
			}

			for (int i = 0; i < beads.Length; i++)
			{
				int bead=beads[i];

				Series xs = checkX.Checked ? new Series("Bead " + bead.ToString() + " X") { ChartType = SeriesChartType.FastLine } : null;
				Series ys = checkY.Checked ? new Series("Bead " + bead.ToString() + " Y") { ChartType = SeriesChartType.FastLine } : null;
				Series zs = checkZ.Checked ? new Series("Bead " + bead.ToString() + " Z") { ChartType = SeriesChartType.FastLine } : null;

				for (int j = 0; j < data.Length; j++)
				{
					var pos = data[j].positions[bead];

					if (refBead>=0) {
						Vec3 refpos=data[j].positions[refBead];
						pos.x-=refpos.x;
						pos.y-=refpos.y;
						pos.z-=refpos.z;
					}

					if (xs != null) xs.Points.AddY(pos.x);
					if (ys != null) ys.Points.AddY(pos.y);
					if (zs != null) zs.Points.AddY(pos.z);
				}

				if (xs != null) chart.Series.Add(xs);
				if (ys != null) chart.Series.Add(ys);
				if (zs != null) chart.Series.Add(zs);
			}
			if (checkMagnetZ.Checked)
			{
				Series mzs = new Series("Magnet Z") { ChartType = SeriesChartType.FastLine };
				chart.Series.Add(mzs);
				for (int i = 0; i < data.Length; i++)
					mzs.Points.AddY(data[i].frameInfo[0]);
			}
			chart.ResetAutoValues();
			chart.ResumeLayout();
			chart.Update();

			/*
			if (filename == null)
				return;
			

			using (FileStream s = File.OpenRead(filename))
			{
				int nf = int.Parse(textNumFramesInView.Text);
				Frame[] data = ReadFrames(s, trackBar.Value, nf);
				chart.Series.Clear();
				int refBead=-1;
				if (txtRefBead.Text.Length >0) {
					refBead=int.Parse(txtRefBead.Text);
				}
				
				for (int i = 0; i < beads.Length; i++)
				{
					var series = chart.Series.Add("Bead " + i.ToString());
					series.ChartType = SeriesChartType.Line;

					for (int j = 0; j < data.Length; j++)
					{
						var pos = data[j].positions;
						double v;
						if (refBead >= 0) v = pos[beads[i]].z - pos[refBead].z;
						else v = pos[beads[i]].z;

						series.Points.AddY(v);
					}

				}
				chart.Update();
			}*/
		}

		private Frame[] ReadFrames(FileStream s, int start, int count)
		{
			if (start < 0) start = 0;
			if (count + start > numFrames)
				count = numFrames - start;
			if (count == 0)
				return new Frame[0];

			s.Seek(startOffset + bytesPerFrame * start, SeekOrigin.Begin);
			Frame[] data = new Frame[count];

			using(BinaryReader r = new BinaryReader(s))
			{
				for (int i = 0; i < count; i++)
					data[i] = ReadFrame(r);
			}
			return data;
		}

		private void trackBar_ValueChanged(object sender, EventArgs e)
		{
			UpdateGraph();
		}


		private void trackBar_Scroll(object sender, EventArgs e)
		{
			labelFrame.Text = trackBar.Value.ToString();
		}
		void UpdateLUT()
		{
			string dirpath = Path.GetDirectoryName(filename);

			if (beadListView.SelectedIndices.Count == 0)
			{
				lutView.Visible = false;
				return;
			}
			lutView.Visible = true;

			string lutimg = dirpath + string.Format("\\lut\\lut{0:D3}", beadListView.SelectedIndices[0]) + ".png";

			try
			{
				Image img = Image.FromFile(lutimg);
				lutView.Image = img;
			}
			catch (Exception e)
			{
				Console.WriteLine("Failed to load {0}. Exception: {1}", lutimg, e.Message);
			}
		}

		private void writeAsTextToolStripMenuItem_Click(object sender, EventArgs e)
		{
			var sfd = new SaveFileDialog();
			if (sfd.ShowDialog() == System.Windows.Forms.DialogResult.OK) {

				Stream txtfile = sfd.OpenFile();

				using (FileStream s = File.OpenRead(filename)) {
				}
			}

		}

		private void repairBinaryToolStripMenuItem_Click(object sender, EventArgs e)
		{
			FileStream s = File.Open(filename, FileMode.Open, FileAccess.ReadWrite);

			s.Seek(0, SeekOrigin.Begin);
			using (BinaryWriter w = new BinaryWriter(s))
			{
				w.Write((int)2); // version
				w.Write(numBeads);
				w.Write(infoColNames.Length);
				w.Write(startOffset);

			}
		}

		private void openOldVersionFileToolStripMenuItem_Click(object sender, EventArgs e)
		{

			OpenFileDialog ofd = new OpenFileDialog();
			if (ofd.ShowDialog() == System.Windows.Forms.DialogResult.OK)
			{
				filename = ofd.FileName;

				ReadHeader(true);
				UpdateGraph();
			}
		}

		void WriteFrameTextData(string fn, Func<Frame, int, float> sel)
		{
			using (var f = File.OpenWrite(fn))
			{
				using (StreamWriter w = new StreamWriter(f))
				{
					for (int i = 0; i < traces[0].frames.Count; i++)
					{
						Frame fr = traces[0].frames[i];
						w.Write(fr.timestamp.ToString(CultureInfo.InvariantCulture));
						w.Write("\t");

						for (int j = 0; j < fr.positions.Length; j++)
						{
							float v = sel(fr, j);
							w.Write(v.ToString(CultureInfo.InvariantCulture));
							if (j < fr.positions.Length - 1) w.Write("\t");
						}

						w.WriteLine();
					}
				}
			}

		}

		private void exportTraces_Click(object sender, EventArgs e)
		{
			SaveFileDialog sfd = new SaveFileDialog() {
				Title = "Select base filename *.txt",
				Filter = "*.txt|*.txt"
			};

			if (sfd.ShowDialog() == System.Windows.Forms.DialogResult.OK)
			{
				string fn = sfd.FileName;
				string fnne = Path.GetDirectoryName(fn) + "\\" + Path.GetFileNameWithoutExtension(fn);

				WriteFrameTextData(fnne + ".z.txt", (fr, i) => fr.positions[i].z);
				WriteFrameTextData(fnne + ".x.txt", (fr, i) => fr.positions[i].x);
				WriteFrameTextData(fnne + ".y.txt", (fr, i) => fr.positions[i].y);
			}
		}

		private void checkUseRef_CheckedChanged(object sender, EventArgs e)
		{
			UpdateGraph();
		}

		private void textNumFramesInView_TextChanged(object sender, EventArgs e)
		{
			UpdateGraph();
		}

		private void checkY_CheckedChanged(object sender, EventArgs e)
		{
			UpdateGraph();

		}

		private void checkMagnetZ_CheckedChanged(object sender, EventArgs e)
		{
			UpdateGraph();
		}

		private void buttonNoiseEstim_Click(object sender, EventArgs e)
		{



		}

		private Vec3[] GetBeadData(Trace tr, int bead, int refBead = -1)
		{
			Vec3[] data = new Vec3[tr.frames.Count];
			for (int k = 0; k < tr.frames.Count; k++)
			{
				data[k] = tr.frames[k].positions[bead];
				if (refBead >= 0)
				{
					Vec3 refbeadpos = tr.frames[k].positions[refBead];
					data[k] = new Vec3() { x = data[k].x - refbeadpos.x, y = data[k].y - refbeadpos.y, z = data[k].z - refbeadpos.z };
				}
			}
			return data;
		}

		float[] MovingAverage(float[] d, int window)
		{
			double[] s = new double[d.Length];
			double sum=0;
			for (int i = 0; i < d.Length; i++)
			{
				s[i] = sum;
				sum += d[i];
			}

			float []ma = new float[d.Length];

			Debug.Assert(window>0);

			for (int i = 0; i < d.Length; i++)
			{
				int start = i - window/2;
				int endpos = start + window;

				if (endpos > d.Length) endpos=d.Length;
				if (start< 0) start=0;

				ma[i] = (float) ( (s[endpos - 1] - s[start]) / window );
			}

			return ma;
		}

		float StandardDeviation(IEnumerable<float> d)
		{
			double sum2=0,sum=0;

			int c = 0;
			foreach(float v in d)
			{
				sum2 += v * v;
				sum += v;
				c++;
			}

			double mean=sum/c;
			return (float)Math.Sqrt(sum2 / (double)c - mean * mean);
		}

		float Noise(float[] d, int smoothWindow, out float[] ma)
		{
			ma = MovingAverage(d, smoothWindow);

			for (int i=0;i<ma.Length;i++)
				ma[i]-=d[i];

			return StandardDeviation(ma);
		}

		int[] BeadSelection
		{
			get
			{
				int[] r = new int[beadListView.CheckedItems.Count];
				for (int i = 0; i < r.Length; i++)
					r[i] = beadListView.CheckedIndices[i];
				return r;
			}
			set
			{
				foreach (ListViewItem item in beadListView.CheckedItems)
					item.Checked = false;

				foreach (int i in value)
					beadListView.Items[i].Checked = true;
			}
		}


		private void beadSelectionToolStripMenuItem_Click(object sender, EventArgs e)
		{
			int refBead;
			if (!int.TryParse(txtRefBead.Text, out refBead))
				refBead = -1;
//			Vec3[] data = GetBeadData(traces[0], beadSelect.Value, refBead);

			var dlg= new FilterDlg(beadInfo, traces[0], BeadSelection);

			if (dlg.ShowDialog() == System.Windows.Forms.DialogResult.OK)
			{
				BeadSelection = dlg.Selection;
			}

		}

		private void txtRefBead_Leave(object sender, EventArgs e)
		{

		}

		private void txtRefBead_Validating(object sender, CancelEventArgs e)
		{

			ResetBeadList();
			UpdateGraph();
		}

		private void txtRefBead_KeyDown(object sender, KeyEventArgs e)
		{
			if (e.KeyCode == Keys.Enter)
			{
				ResetBeadList();
				UpdateGraph();
			}
		}

		private void beadListView_SelectedIndexChanged(object sender, EventArgs e)
		{
			UpdateGraph();
			UpdateLUT();
		}

		private void exportBeadSelectionToTxtToolStripMenuItem_Click(object sender, EventArgs e)
		{
			var sfd=new SaveFileDialog() {
			};

			if (sfd.ShowDialog() == System.Windows.Forms.DialogResult.OK)
			{
				using (var stream = sfd.OpenFile())
					using (StreamWriter w = new StreamWriter(stream))
					{
						int[] sel = BeadSelection;
						for (int i = 0; i < sel.Length; i++)
						{
							w.Write(sel[i]);
							if (i < sel.Length - 1) w.WriteLine();
						}
					}
			}
		}

	}
}
