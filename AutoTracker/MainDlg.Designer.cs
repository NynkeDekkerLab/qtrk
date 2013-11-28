namespace AutoTracker
{
    partial class AutoTrackerDlg
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
					this.components = new System.ComponentModel.Container();
					this.menuStrip1 = new System.Windows.Forms.MenuStrip();
					this.openToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
					this.splitContainer1 = new System.Windows.Forms.SplitContainer();
					this.lblStepYPos = new System.Windows.Forms.Label();
					this.lblStepXPos = new System.Windows.Forms.Label();
					this.label2 = new System.Windows.Forms.Label();
					this.label1 = new System.Windows.Forms.Label();
					this.timerUIUpdate = new System.Windows.Forms.Timer(this.components);
					this.menuStrip1.SuspendLayout();
					((System.ComponentModel.ISupportInitialize)(this.splitContainer1)).BeginInit();
					this.splitContainer1.Panel1.SuspendLayout();
					this.splitContainer1.SuspendLayout();
					this.SuspendLayout();
					// 
					// menuStrip1
					// 
					this.menuStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.openToolStripMenuItem});
					this.menuStrip1.Location = new System.Drawing.Point(0, 0);
					this.menuStrip1.Name = "menuStrip1";
					this.menuStrip1.Size = new System.Drawing.Size(379, 24);
					this.menuStrip1.TabIndex = 0;
					this.menuStrip1.Text = "menuStrip1";
					// 
					// openToolStripMenuItem
					// 
					this.openToolStripMenuItem.Name = "openToolStripMenuItem";
					this.openToolStripMenuItem.Size = new System.Drawing.Size(48, 20);
					this.openToolStripMenuItem.Text = "Open";
					// 
					// splitContainer1
					// 
					this.splitContainer1.Dock = System.Windows.Forms.DockStyle.Fill;
					this.splitContainer1.Location = new System.Drawing.Point(0, 24);
					this.splitContainer1.Name = "splitContainer1";
					// 
					// splitContainer1.Panel1
					// 
					this.splitContainer1.Panel1.Controls.Add(this.lblStepYPos);
					this.splitContainer1.Panel1.Controls.Add(this.label1);
					this.splitContainer1.Panel1.Controls.Add(this.lblStepXPos);
					this.splitContainer1.Panel1.Controls.Add(this.label2);
					this.splitContainer1.Size = new System.Drawing.Size(379, 238);
					this.splitContainer1.SplitterDistance = 125;
					this.splitContainer1.TabIndex = 1;
					// 
					// lblStepYPos
					// 
					this.lblStepYPos.AutoSize = true;
					this.lblStepYPos.Location = new System.Drawing.Point(65, 91);
					this.lblStepYPos.Name = "lblStepYPos";
					this.lblStepYPos.Size = new System.Drawing.Size(0, 13);
					this.lblStepYPos.TabIndex = 1;
					// 
					// lblStepXPos
					// 
					this.lblStepXPos.AutoSize = true;
					this.lblStepXPos.Location = new System.Drawing.Point(65, 59);
					this.lblStepXPos.Name = "lblStepXPos";
					this.lblStepXPos.Size = new System.Drawing.Size(0, 13);
					this.lblStepXPos.TabIndex = 1;
					// 
					// label2
					// 
					this.label2.Location = new System.Drawing.Point(12, 59);
					this.label2.Name = "label2";
					this.label2.Size = new System.Drawing.Size(100, 23);
					this.label2.TabIndex = 2;
					// 
					// label1
					// 
					this.label1.AutoSize = true;
					this.label1.Location = new System.Drawing.Point(24, 10);
					this.label1.Name = "label1";
					this.label1.Size = new System.Drawing.Size(57, 13);
					this.label1.TabIndex = 0;
					this.label1.Text = "Stepper X:";
					// 
					// timerUIUpdate
					// 
					this.timerUIUpdate.Tick += new System.EventHandler(this.timerUIUpdate_Tick);
					// 
					// AutoTrackerDlg
					// 
					this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
					this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
					this.ClientSize = new System.Drawing.Size(379, 262);
					this.Controls.Add(this.splitContainer1);
					this.Controls.Add(this.menuStrip1);
					this.Name = "AutoTrackerDlg";
					this.Text = "StatusDlg";
					this.FormClosing += new System.Windows.Forms.FormClosingEventHandler(this.AutoTrackerDlg_FormClosing);
					this.menuStrip1.ResumeLayout(false);
					this.menuStrip1.PerformLayout();
					this.splitContainer1.Panel1.ResumeLayout(false);
					this.splitContainer1.Panel1.PerformLayout();
					((System.ComponentModel.ISupportInitialize)(this.splitContainer1)).EndInit();
					this.splitContainer1.ResumeLayout(false);
					this.ResumeLayout(false);
					this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.MenuStrip menuStrip1;
        private System.Windows.Forms.SplitContainer splitContainer1;
        private System.Windows.Forms.ToolStripMenuItem openToolStripMenuItem;
				private System.Windows.Forms.Label label2;
				private System.Windows.Forms.Label label1;
				private System.Windows.Forms.Label lblStepYPos;
				private System.Windows.Forms.Label lblStepXPos;
				private System.Windows.Forms.Timer timerUIUpdate;

    }
}