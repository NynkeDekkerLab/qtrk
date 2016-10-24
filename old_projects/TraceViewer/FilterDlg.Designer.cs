namespace TraceViewer
{
	partial class FilterDlg
	{
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.IContainer components = null;

		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		/// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
		protected override void Dispose(bool disposing)
		{
			if (disposing && (components != null))
			{
				components.Dispose();
			}
			base.Dispose(disposing);
		}

		#region Windows Form Designer generated code

		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		private void InitializeComponent()
		{
			this.textMaxNoiseValue = new System.Windows.Forms.TextBox();
			this.labelMaxNoise = new System.Windows.Forms.Label();
			this.label1 = new System.Windows.Forms.Label();
			this.labelNumValidBeads = new System.Windows.Forms.Label();
			this.label2 = new System.Windows.Forms.Label();
			this.labelNumTotalBeads = new System.Windows.Forms.Label();
			this.buttonApply = new System.Windows.Forms.Button();
			this.checkFilterCurrent = new System.Windows.Forms.CheckBox();
			this.buttonCancel = new System.Windows.Forms.Button();
			this.textMinTetherLen = new System.Windows.Forms.TextBox();
			this.label3 = new System.Windows.Forms.Label();
			this.SuspendLayout();
			// 
			// textMaxNoiseValue
			// 
			this.textMaxNoiseValue.Location = new System.Drawing.Point(128, 12);
			this.textMaxNoiseValue.Name = "textMaxNoiseValue";
			this.textMaxNoiseValue.Size = new System.Drawing.Size(100, 20);
			this.textMaxNoiseValue.TabIndex = 0;
			this.textMaxNoiseValue.Text = "0,05";
			this.textMaxNoiseValue.TextChanged += new System.EventHandler(this.textMaxNoiseValue_TextChanged);
			// 
			// labelMaxNoise
			// 
			this.labelMaxNoise.AutoSize = true;
			this.labelMaxNoise.Location = new System.Drawing.Point(12, 12);
			this.labelMaxNoise.Name = "labelMaxNoise";
			this.labelMaxNoise.Size = new System.Drawing.Size(87, 13);
			this.labelMaxNoise.TabIndex = 1;
			this.labelMaxNoise.Text = "Max noise value:";
			// 
			// label1
			// 
			this.label1.AutoSize = true;
			this.label1.Location = new System.Drawing.Point(269, 50);
			this.label1.Name = "label1";
			this.label1.Size = new System.Drawing.Size(72, 13);
			this.label1.TabIndex = 2;
			this.label1.Text = "#Valid beads:";
			// 
			// labelNumValidBeads
			// 
			this.labelNumValidBeads.AutoSize = true;
			this.labelNumValidBeads.Location = new System.Drawing.Point(360, 50);
			this.labelNumValidBeads.Name = "labelNumValidBeads";
			this.labelNumValidBeads.Size = new System.Drawing.Size(0, 13);
			this.labelNumValidBeads.TabIndex = 2;
			// 
			// label2
			// 
			this.label2.AutoSize = true;
			this.label2.Location = new System.Drawing.Point(269, 27);
			this.label2.Name = "label2";
			this.label2.Size = new System.Drawing.Size(73, 13);
			this.label2.TabIndex = 2;
			this.label2.Text = "#Total beads:";
			// 
			// labelNumTotalBeads
			// 
			this.labelNumTotalBeads.AutoSize = true;
			this.labelNumTotalBeads.Location = new System.Drawing.Point(360, 27);
			this.labelNumTotalBeads.Name = "labelNumTotalBeads";
			this.labelNumTotalBeads.Size = new System.Drawing.Size(0, 13);
			this.labelNumTotalBeads.TabIndex = 2;
			// 
			// buttonApply
			// 
			this.buttonApply.Location = new System.Drawing.Point(16, 132);
			this.buttonApply.Name = "buttonApply";
			this.buttonApply.Size = new System.Drawing.Size(75, 23);
			this.buttonApply.TabIndex = 3;
			this.buttonApply.Text = "Apply";
			this.buttonApply.UseVisualStyleBackColor = true;
			this.buttonApply.Click += new System.EventHandler(this.buttonApply_Click);
			// 
			// checkFilterCurrent
			// 
			this.checkFilterCurrent.AutoSize = true;
			this.checkFilterCurrent.Location = new System.Drawing.Point(17, 38);
			this.checkFilterCurrent.Name = "checkFilterCurrent";
			this.checkFilterCurrent.Size = new System.Drawing.Size(129, 17);
			this.checkFilterCurrent.TabIndex = 4;
			this.checkFilterCurrent.Text = "Filter current selection";
			this.checkFilterCurrent.UseVisualStyleBackColor = true;
			this.checkFilterCurrent.CheckedChanged += new System.EventHandler(this.checkFilterCurrent_CheckedChanged);
			// 
			// buttonCancel
			// 
			this.buttonCancel.Location = new System.Drawing.Point(97, 132);
			this.buttonCancel.Name = "buttonCancel";
			this.buttonCancel.Size = new System.Drawing.Size(75, 23);
			this.buttonCancel.TabIndex = 3;
			this.buttonCancel.Text = "Cancel";
			this.buttonCancel.UseVisualStyleBackColor = true;
			this.buttonCancel.Click += new System.EventHandler(this.buttonApply_Click);
			// 
			// textMinTetherLen
			// 
			this.textMinTetherLen.Location = new System.Drawing.Point(128, 61);
			this.textMinTetherLen.Name = "textMinTetherLen";
			this.textMinTetherLen.Size = new System.Drawing.Size(100, 20);
			this.textMinTetherLen.TabIndex = 0;
			this.textMinTetherLen.Text = "1";
			this.textMinTetherLen.TextChanged += new System.EventHandler(this.textMaxNoiseValue_TextChanged);
			// 
			// label3
			// 
			this.label3.AutoSize = true;
			this.label3.Location = new System.Drawing.Point(14, 64);
			this.label3.Name = "label3";
			this.label3.Size = new System.Drawing.Size(74, 13);
			this.label3.TabIndex = 1;
			this.label3.Text = "Min tether len:";
			// 
			// FilterDlg
			// 
			this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
			this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
			this.ClientSize = new System.Drawing.Size(527, 257);
			this.Controls.Add(this.checkFilterCurrent);
			this.Controls.Add(this.buttonCancel);
			this.Controls.Add(this.buttonApply);
			this.Controls.Add(this.labelNumValidBeads);
			this.Controls.Add(this.labelNumTotalBeads);
			this.Controls.Add(this.label2);
			this.Controls.Add(this.label1);
			this.Controls.Add(this.label3);
			this.Controls.Add(this.labelMaxNoise);
			this.Controls.Add(this.textMinTetherLen);
			this.Controls.Add(this.textMaxNoiseValue);
			this.Name = "FilterDlg";
			this.Text = "FilterDlg";
			this.ResumeLayout(false);
			this.PerformLayout();

		}

		#endregion

		private System.Windows.Forms.TextBox textMaxNoiseValue;
		private System.Windows.Forms.Label labelMaxNoise;
		private System.Windows.Forms.Label label1;
		private System.Windows.Forms.Label labelNumValidBeads;
		private System.Windows.Forms.Label label2;
		private System.Windows.Forms.Label labelNumTotalBeads;
		private System.Windows.Forms.Button buttonApply;
		private System.Windows.Forms.CheckBox checkFilterCurrent;
		private System.Windows.Forms.Button buttonCancel;
		private System.Windows.Forms.TextBox textMinTetherLen;
		private System.Windows.Forms.Label label3;
	}
}