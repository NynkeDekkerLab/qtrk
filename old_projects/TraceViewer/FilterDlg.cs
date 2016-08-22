using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;

namespace TraceViewer
{
	public partial class FilterDlg : Form
	{

		ViewTraceDlg.BeadInfo[] beadInfo;
		ViewTraceDlg.Trace trace;
		int[] currentSelection;
		int[] filtered;

		public FilterDlg(ViewTraceDlg.BeadInfo[] beadInfo, ViewTraceDlg.Trace trace, int[] selection)
		{
			InitializeComponent();

			this.beadInfo = beadInfo;
			this.trace = trace;
			this.currentSelection = selection;


			UpdateFilter();
		}

		private void UpdateFilter()
		{
			int [] selection = currentSelection;

			if (!checkFilterCurrent.Checked) {
				selection = new int[beadInfo.Length].Select( (v,i)=>i ).ToArray();
			}

			List<int> valid=new List<int>(selection.Length);
			float maxNoise=0;
			float.TryParse(textMaxNoiseValue.Text, out maxNoise);

			float minTetherLen = 0;
			float.TryParse(textMinTetherLen.Text, out minTetherLen);

			for (int i=0;i<selection.Length;i++) {
				int bead=selection[i];

				float len = beadInfo[bead].max - beadInfo[bead].min;

				if (beadInfo[bead].noise<=maxNoise && len >= minTetherLen)
					valid.Add(bead);
			}

			labelNumTotalBeads.Text = selection.Length.ToString();
			labelNumValidBeads.Text = valid.Count.ToString();

			filtered = valid.ToArray();
		}

		private void buttonApply_Click(object sender, EventArgs e)
		{
			DialogResult = System.Windows.Forms.DialogResult.OK;
		}

		public int[] Selection { get { return filtered; } }

		private void textMaxNoiseValue_TextChanged(object sender, EventArgs e)
		{
			UpdateFilter();
		}

		private void checkFilterCurrent_CheckedChanged(object sender, EventArgs e)
		{
			UpdateFilter();
		}
	}
}
